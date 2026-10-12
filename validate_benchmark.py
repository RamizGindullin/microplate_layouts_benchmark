#!/usr/bin/env python3
"""Validate benchmark inputs and reference cases without a full benchmark run."""
import argparse
import contextlib
import hashlib
import importlib
import importlib.metadata
import io
import json
import os
from pathlib import Path, PurePosixPath
import platform
import re
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from unittest.mock import patch
import zipfile


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def git_info(root):
    def command(*args):
        result = subprocess.run(
            ['git', '-C', str(root), *args], capture_output=True, text=True,
            timeout=10, check=False,
        )
        return result.stdout.strip() if result.returncode == 0 else None
    try:
        head = command('rev-parse', 'HEAD')
        status = command('status', '--porcelain')
        return {'commit': head, 'working_tree_dirty': bool(status) if status is not None else None}
    except (OSError, subprocess.TimeoutExpired):
        return {'commit': None, 'working_tree_dirty': None}


class LayoutReader:
    def __init__(self, root, np, archive_only=False):
        self.root = root
        self.np = np
        self.archive_only = archive_only
        self.archive = root / 'layouts.zip'
        self.zip = zipfile.ZipFile(self.archive) if self.archive.is_file() else None
        self.cache = {}
        self.manifest = {}

    def files(self, directory, regex):
        folder = Path(directory)
        local = folder if folder.is_absolute() else self.root / folder
        if local.is_dir() and not self.archive_only:
            return [
                ('file', str(p)) for p in sorted(local.iterdir())
                if p.is_file() and p.suffix == '.npy' and re.search(regex, p.name)
            ]
        if self.zip is None:
            raise FileNotFoundError(f'Neither usable directory {local} nor layouts.zip found')
        logical = str(folder).replace('\\', '/').strip('/')
        basename = PurePosixPath(logical).name
        members = []
        for name in sorted(self.zip.namelist()):
            p = PurePosixPath(name)
            parent = str(p.parent)
            matches_dir = (parent == logical or parent.endswith('/' + logical)
                           or parent == basename or parent.endswith('/' + basename))
            if matches_dir and p.suffix == '.npy' and re.search(regex, p.name):
                members.append(('zip', name))
        parents = {str(PurePosixPath(name).parent) for _, name in members}
        require(len(parents) <= 1, f'Ambiguous archive directories for {directory}: {sorted(parents)}')
        return members

    def load(self, entry):
        kind, name = entry
        identifier = str(Path(name).resolve()) if kind == 'file' else 'layouts.zip::' + name
        if identifier not in self.cache:
            data = Path(name).read_bytes() if kind == 'file' else self.zip.read(name)
            array = self.np.load(io.BytesIO(data), allow_pickle=False)
            self.cache[identifier] = array
            self.manifest[identifier] = {'sha256': sha256(data), 'shape': list(array.shape), 'dtype': str(array.dtype)}
        return self.cache[identifier].copy()

    def close(self):
        if self.zip is not None:
            self.zip.close()


def validate_layout(np, layout, n_compound_slots, negative_count, positive_count=0, repeats=1):
    require(layout.shape == (16, 24), f'Expected shape (16, 24), got {layout.shape}')
    require(np.issubdtype(layout.dtype, np.number), f'Non-numeric layout dtype {layout.dtype}')
    require(not np.iscomplexobj(layout), 'Complex identifiers are not supported')
    require(np.all(np.isfinite(layout)), 'Non-finite identifiers')
    require(np.all(layout == np.floor(layout)), 'Non-integer-valued identifiers')
    require(np.all(layout >= 0), 'Negative identifiers')
    require(np.all(layout[[0, -1], :] == 0) and np.all(layout[:, [0, -1]] == 0), 'Nonempty buffer border')
    negative_id = n_compound_slots + (2 if positive_count else 1)
    require(int(np.max(layout)) == negative_id, f'Expected negative-control ID {negative_id}, got {np.max(layout)}')
    require(np.count_nonzero(layout == negative_id) == negative_count, 'Wrong negative-control count')
    if positive_count:
        require(np.count_nonzero(layout == negative_id - 1) == positive_count, 'Wrong positive-control count')
    mask = (layout > 0) & (layout <= n_compound_slots)
    ids, counts = np.unique(layout[mask], return_counts=True)
    require(np.array_equal(ids, np.arange(1, n_compound_slots + 1)), 'Missing or unexpected experimental IDs')
    require(np.all(counts == repeats), f'Expected {repeats} occurrences of each experimental ID')
    require(np.count_nonzero(layout) == n_compound_slots * repeats + negative_count + positive_count, 'Unexpected occupied-well count')


def run(args):
    root = Path(args.root).resolve() if args.root else Path(__file__).resolve().parent
    report_path = Path(args.report).resolve() if args.report else root / 'validation_report.json'
    report = {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'scope': 'unit-only' if args.unit_only else 'registered layouts and reference cases',
        'root': str(root), 'git': git_info(root),
        'environment': {'python': sys.version, 'platform': platform.platform()},
        'tests': [], 'layout_groups': [], 'layout_manifest': {},
        'limitations': [
            'Only layouts selected by the current default registries and configuration are audited.',
            'Synthetic fixtures verify the implemented ID contract, not external generator intentions or optimization constraints.',
            'This script does not audit the full simulation CSV row counts or establish full-run reproducibility.',
            'A successful validation run is not proof of biological realism or correctness of every numerical output.',
        ],
    }
    for package in ('numpy', 'pandas', 'scipy', 'scikit-learn', 'matplotlib', 'loess', 'moepy'):
        try:
            report['environment'][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            report['environment'][package] = None

    def check(name, operation):
        try:
            details = operation()
            record = {'name': name, 'status': 'PASS'}
            if details is not None:
                record['details'] = details
        except Exception as exc:
            record = {'name': name, 'status': 'FAIL', 'error_type': type(exc).__name__,
                      'message': str(exc), 'traceback': traceback.format_exc()}
        report['tests'].append(record)
        print(f"[{record['status']}] {name}" + (f": {record['message']}" if record['status'] == 'FAIL' else ''))

    reader = None
    old_cwd = Path.cwd()
    try:
        require((root / 'benchmark_common.py').is_file(), f'Not a benchmark root: {root}')
        sys.path.insert(0, str(root))
        os.chdir(root)
        os.environ.setdefault('MPLBACKEND', 'Agg')
        import numpy as np
        import pandas as pd
        import benchmark_common as common
        import benchmark_disturbances as registry
        import libraries.dose_response as dr
        import libraries.disturbances as dt
        import libraries.normalization as norm
        import run_dose_response_benchmark as dr_driver
        import run_screening_benchmark as hts_driver
        for module in (common, registry, dr, dt, norm, dr_driver, hts_driver):
            path = Path(module.__file__).resolve()
            require(path.is_relative_to(root), f'Imported a module outside the requested repository: {path}')
        report['source_sha256'] = {
            str(Path(m.__file__).resolve().relative_to(root)): sha256(Path(m.__file__).read_bytes())
            for m in (common, registry, dr, dt, norm, dr_driver, hts_driver)
        }

        params = [
            {'compound': 0, 'b': 1., 'c': 0., 'd': 100., 'e': 2., 'startDose': 4., 'nDose': 2, 'dilution': 2.},
            {'compound': 1, 'b': 1., 'c': 0., 'd': 100., 'e': 3., 'startDose': 4., 'nDose': 2, 'dilution': 2.},
        ]
        # Independent, hand-labelled order: (replicate, compound, dose, marker).
        oracle = [(0,0,4.,11.), (0,0,2.,12.), (0,1,4.,21.), (0,1,2.,22.),
                  (1,0,4.,111.), (1,0,2.,112.), (1,1,4.,121.), (1,1,2.,122.)]
        marker_by_id = {1:122., 2:121., 3:112., 4:111., 5:22., 6:21., 7:12., 8:11.}
        final = np.zeros((16,24), dtype=int)
        for row, ids in [(1,[1,2,3,4]), (2,[5,6,7,8])]:
            final[row,1:5] = ids
        final[1,6] = final[2,6] = 9
        legacy = final.copy()
        legacy[2,1:5] = [1,2,3,4]
        legacy[legacy == 9] = 5
        markers = np.array([row[3] for row in oracle])
        content = pd.DataFrame({'compound':[row[1] for row in oracle],
                                'dose':[row[2] for row in oracle], 'response':markers})
        r, j = np.indices((16,24))
        bowl_factor = (1 + .055 * np.abs(r - 7.5)) * (1 + .055 * np.abs(j - 11.5))

        def generated_order():
            generated = dr.generate_plate_content(params, replicates=2)
            np.testing.assert_array_equal(generated['compound'].to_numpy(), content['compound'].to_numpy())
            np.testing.assert_array_equal(generated['dose'].to_numpy(), content['dose'].to_numpy())
            expected = np.array([100. / (1. + row[2] / params[row[1]]['e']) for row in oracle])
            np.testing.assert_allclose(generated['response'].to_numpy(), expected, rtol=1e-12, atol=1e-12)
        check('Synthetic observation order and response generation', generated_order)

        def conversion():
            converted = dr.update_compd_layout(legacy, 2, 2, 2)
            np.testing.assert_array_equal(converted, final)
        check('Legacy conversion: independently specified per-well IDs', conversion)

        def placement():
            plate = dr.fill_plate(final, content.copy(), neg_control_value=100, expected_noise=0.)
            for identifier, marker in marker_by_id.items():
                np.testing.assert_array_equal(plate[final == identifier], [marker])
            np.testing.assert_array_equal(plate[final == 9], [100.,100.])
            np.testing.assert_array_equal(plate[final == 0], 0.)
            np.testing.assert_array_equal(dr.collect_plate_results(final,plate), markers)
        check('Marker placement and result recovery against a hand-labelled oracle', placement)

        def input_guards():
            plate = np.zeros((16,24))
            malformed = [
                (np.array([1,2]), np.zeros(2)), (final, np.zeros((15,24))),
                (np.where(final == 1, np.nan, final), plate),
                (np.where(final == 1, np.inf, final), plate),
                (np.where(final == 1, 1.5, final), plate),
            ]
            for layout, signals in malformed:
                try:
                    dr.collect_plate_results(layout,signals)
                except ValueError:
                    continue
                raise AssertionError('A malformed fixture was not rejected')
        check('Existing DR result-collection guards', input_guards)

        def validator_rejects_duplicate():
            layout = np.zeros((16,24), dtype=int)
            layout[1,1:5] = [1,2,3,4]
            layout[2,1:5] = [5,6,7,8]
            layout[1,6] = layout[2,6] = 9
            validate_layout(np, layout,8,2)
            layout[3,1] = 1
            try:
                validate_layout(np,layout,8,2)
            except AssertionError:
                return
            raise AssertionError('The validation script did not reject a duplicated experimental ID')
        check('Validation audit detects duplicated IDs', validator_rejects_duplicate)

        def independent_disturbances():
            plate = np.zeros((16,24));plate[1:-1,1:-1] = 60.
            cases = [
                (dt.add_bowlshaped_errors_nl, .055, bowl_factor),
                (dt.add_errors_to_upper_rows, .0133, 1 + .0133*(15-r)),
                (dt.add_errors_to_right_columns_half, .2, np.where(j<12,1.,1+.2*(j-12)/11.)),
                (dt.add_striped_errors_pure_even, .2, np.where(r%2==0,1.2,1.)),
            ]
            for function, strength, factor in cases:
                np.testing.assert_allclose(function(plate.copy(),strength),plate*factor,rtol=1e-12,atol=1e-12)
                np.testing.assert_array_equal(function(plate.copy(),0.),plate)
        check('Disturbance formulas and zero-strength identities', independent_disturbances)

        def control_protection():
            plate = np.zeros((16,24));plate[final>0]=60.
            plate[1,6]=99.997;plate[2,6]=100.003
            before = plate.copy()
            affected = dt.apply_disturbance(plate,final,9,dt.add_bowlshaped_errors_nl,.055,neg_controls_affected=True)
            protected = dt.apply_disturbance(plate,final,9,dt.add_bowlshaped_errors_nl,.055,neg_controls_affected=False)
            np.testing.assert_allclose(affected,before*bowl_factor,rtol=1e-12,atol=1e-12)
            np.testing.assert_array_equal(protected[final==9],before[final==9])
            np.testing.assert_array_equal(protected[(final>0)&(final<9)],affected[(final>0)&(final<9)])
            np.testing.assert_array_equal(protected[final==0],0.)
            np.testing.assert_array_equal(plate,before)
        check('Control protection retains measured values, not a fixed 100', control_protection)

        def experiment_path():
            for raw, convert in [(final,False),(legacy,True)]:
                for affected in (False,True):
                    with patch.object(dr.np,'load',return_value=raw.copy()):
                        obtained, controls = dr._run_experiment(
                            layout_dir='', layout_file='fixture.npy', plate_content=content.copy(),
                            expected_noise=0., error_function=dt.add_bowlshaped_errors_nl, error=.055,
                            normalization_function=lambda plate,layout,neg_control_id,min_dist=0:plate.copy(),
                            min_dist=0, lose_from_row=0, lose_to_row=0, neg_controls_affected=affected,
                            plate_type={'requires_layout_update':convert}, compounds=2, concentrations=2, replicates=2,
                        )
                    expected = []
                    for marker in markers:
                        identifier = next(k for k,v in marker_by_id.items() if v==marker)
                        row,col = np.argwhere(final==identifier)[0]
                        expected.append(marker*bowl_factor[row,col])
                    np.testing.assert_allclose(obtained['results'].to_numpy(),expected,rtol=1e-12,atol=1e-12)
                    expected_controls = 100*bowl_factor[final==9] if affected else np.array([100.,100.])
                    np.testing.assert_allclose(controls,expected_controls,rtol=1e-12,atol=1e-12)
        check('Actual experiment path: mapping, conversion and control-flag propagation', experiment_path)

        def scenario_flags():
            scenarios = registry.dr_scenarios()
            protected = [s.key for s in scenarios if not s.dr_neg_controls_affected]
            require(protected == ['bowl_nl_neg_unaffected'], f'Unexpected protected scenarios: {protected}')
            flags = {s.dr_id_text:s.dr_neg_controls_affected for s in scenarios}
            for scenario in dr_driver._build_default_dr_scenarios():
                for et in scenario.error_types:
                    require(et['neg_controls_affected']==flags[scenario.id_text], 'Scenario builder lost the control flag')
        check('Registry and scenario-builder control flags', scenario_flags)

        def main_call_forwarding():
            true = pd.DataFrame({'b':[1.,1.], 'c':[0.,0.], 'd':[100.,100.], 'e':[10.,20.]})
            true['abs IC50'] = [10.,20.]
            fitted = true[['b','c','d','e']].copy()
            fitted['r2_score'] = 1.
            fitted['residuals'] = [np.zeros(2),np.zeros(2)]
            fitted['true_residuals'] = [np.zeros(2),np.zeros(2)]
            correction = lambda *args,**kwargs:args[0]
            for affected in (False,True):
                with patch.object(dr,'plate_curves_after_error',return_value=fitted.copy()) as mocked:
                    dr._run_one_plate(
                        plate_type={'dir':'','display_type':'Fixture','error_correction':correction},
                        layout_file='fixture.npy',plate_content=content.copy(),
                        et={'type':'fixture','error_function':dt.add_bowlshaped_errors_nl,'error':.055,'neg_controls_affected':affected},
                        current_e=1,compounds=2,concentrations=2,replicates=2,
                        limit={'from':0,'to':0},df_params=true,expected_noise=0.,
                    )
                require(mocked.call_args.kwargs['neg_controls_affected'] is affected, 'Main plate runner lost the control flag')
                require(mocked.call_args.args[6] is correction, 'Main runner did not use the layout correction')
        check('Main plate runner selects layout correction and forwards control treatment', main_call_forwarding)

        def analytic_ic50():
            count = 0
            for b in (.5,1.,1.5):
                for lower in (0.,10.,25.):
                    for upper in (75.,100.):
                        for midpoint in (1.,10.,50.):
                            expected = midpoint*((upper-50.)/(50.-lower))**(1./b)
                            actual = float(dr.IC50(b,lower,upper,midpoint))
                            np.testing.assert_allclose(actual,expected,rtol=1e-12,atol=1e-10)
                            np.testing.assert_allclose(dr.ll4(actual,b,lower,upper,midpoint),50.,rtol=1e-12,atol=1e-10)
                            count += 1
            return {'valid_crossing_cases':count,'rtol':1e-12,'atol':1e-10}
        check('Analytical IC50 and LL.4 crossing consistency', analytic_ic50)

        def loess_reference():
            layout = np.zeros((16, 24), dtype=int)
            layout[1:-1, 1:-1] = 1

            for row in (1, 5, 9, 14):
                for col in (1, 6, 11, 16, 22):
                    layout[row, col] = 2

            control_values = (80., 100., 120.)

            for control_value in control_values:
                plate = np.zeros((16, 24))
                plate[layout > 0] = 60.
                plate[layout == 2] = control_value
                before = plate.copy()

                with np.errstate(divide='ignore', invalid='ignore'):
                    normalized = norm.normalize_plate_lowess_2d(
                        plate, layout, 2
                    )

                expected = before * (100. / control_value)

                require(
                    np.all(np.isfinite(normalized)),
                    'Non-finite output on constant-control reference',
                )
                np.testing.assert_allclose(
                    normalized, expected, rtol=1e-8, atol=1e-8
                )
                np.testing.assert_array_equal(plate, before)

            return {
                'constant_control_values': list(control_values),
                'rtol': 1e-8,
                'atol': 1e-8,
            }
        check('LOESS constant-control reference and nonmutation', loess_reference)

        if not args.unit_only:
            reader = LayoutReader(root,np,args.archive_only)
            dr_cfg = dr_driver.DoseResponseConfig()
            hts_cfg = hts_driver.ScreeningConfig()

            def audit_group(label, directory, regex, validator):
                entries = reader.files(directory,regex)
                require(bool(entries), f'No matching .npy layouts: {directory}, regex={regex}')
                failures = []
                for entry in entries:
                    try:
                        validator(reader.load(entry))
                    except Exception as exc:
                        failures.append({'file':entry[1],'error_type':type(exc).__name__,'message':str(exc)})
                details={'group':label,'source':entries[0][0],'matched_files':len(entries),'failed_files':failures}
                report['layout_groups'].append(details)
                require(not failures, json.dumps(failures[:5]))
                return {'matched_files':len(entries),'source':entries[0][0]}

            for doses in dr_cfg.concentrations_list:
                for replicates in dr_cfg.replicates_list:
                    compounds=(14*22-20)//(doses*replicates)
                    for spec in common.DOSE_RESPONSE_LAYOUT_SPECS:
                        label=f'DR layouts: {spec.display_type}, C={compounds}, D={doses}, R={replicates}'
                        regex=spec.regex_template.format(compounds=compounds,concentrations=doses,replicates=replicates)
                        def validator(layout,C=compounds,D=doses,R=replicates,spec=spec):
                            if spec.requires_layout_update:
                                validate_layout(np,layout,C*D,20,repeats=R)
                                layout=dr.update_compd_layout(layout,C,D,R)
                            validate_layout(np,layout,C*D*R,20)
                        check(label,lambda label=label,spec=spec,regex=regex,validator=validator:audit_group(label,spec.layout_dir,regex,validator))
            for negative,positive in hts_cfg.neg_pos_controls_list:
                for spec in common.SCREENING_LAYOUT_SPECS:
                    label=f'HTS layouts: {spec.display_type}, controls=({negative},{positive})'
                    regex=spec.regex_template.format(neg_controls=negative,pos_controls=positive)
                    validator=lambda layout,n=negative,p=positive:validate_layout(np,layout,308-n-p,n,p)
                    check(label,lambda label=label,spec=spec,regex=regex,validator=validator:audit_group(label,spec.layout_dir,regex,validator))
            report['layout_manifest']=reader.manifest
    except Exception as exc:
        report['tests'].append({'name':'Initialization or validation setup','status':'FAIL',
                               'error_type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc()})
        print(f'[FAIL] Initialization or validation setup: {exc}')
    finally:
        if reader is not None:
            report['layout_manifest']=reader.manifest
            reader.close()
        os.chdir(old_cwd)

    passed=sum(t['status']=='PASS' for t in report['tests'])
    failed=sum(t['status']=='FAIL' for t in report['tests'])
    report['summary']={'passed_checks':passed,'failed_checks':failed,
                       'unique_layout_files_loaded':len(report['layout_manifest']),
                       'layout_configuration_groups':len(report['layout_groups']),
                       'layout_audit_requested':not args.unit_only}
    report['status']='FAIL' if failed else 'PASS_UNIT_ONLY' if args.unit_only else 'PASS'
    report_path.parent.mkdir(parents=True,exist_ok=True)
    report_path.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(f"\n{report['status']}: {passed} passed; {failed} failed; {len(report['layout_manifest'])} unique layout files loaded")
    print(f'Report: {report_path}')
    return 1 if failed else 0


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',help='Repository root; defaults to the script directory')
    parser.add_argument('--report',help='JSON report path; defaults to validation_report.json in the repository root')
    parser.add_argument('--unit-only',action='store_true',help='Run reference cases only; do not claim an archive/layout audit')
    parser.add_argument('--archive-only',action='store_true',help='Audit layouts.zip even if extracted layout directories exist')
    args=parser.parse_args()
    if args.unit_only and args.archive_only:
        parser.error('--unit-only and --archive-only cannot be combined')
    return run(args)


if __name__=='__main__':
    sys.exit(main())
