import numpy as np
import math

def add_bowlshaped_errors(plate, error):
    
    plate_array = __add_bowlshaped_errors_to_columns(plate, error)
    plate_array = __add_bowlshaped_errors_to_rows(plate_array, error)
    
    return plate_array



def __add_bowlshaped_errors_to_columns(plate, error):
    num_rows, num_columns = plate.shape
    plate = np.asarray(plate, dtype=float)
    plate_array = plate.copy()
    
    translation_const = (num_columns - 1) / 2.0
    
    for row_index in range(num_rows):
        for col_index in range(num_columns):
            if (plate[row_index][col_index] > 0):
                plate_array[row_index][col_index] += error*abs(col_index - translation_const)
                plate_array[row_index][col_index] = max(plate_array[row_index][col_index],0.0)
            
    return plate_array



def __add_bowlshaped_errors_to_rows(plate, error):
    num_rows, num_columns = plate.shape
    plate = np.asarray(plate, dtype=float)
    plate_array = plate.copy()

    translation_const = (num_rows - 1) / 2.0
    
    for row_index in range(num_rows):
        for col_index in range(num_columns):
            if (plate[row_index][col_index] > 0):
                plate_array[row_index][col_index] += error*abs(row_index - translation_const)
                plate_array[row_index][col_index] = max(plate_array[row_index][col_index],0.0)
                
    return plate_array


def add_bowlshaped_errors_nl(plate, error=0.125):
    
    plate_array = __add_bowlshaped_errors_to_columns_nl(plate, error)
    plate_array = __add_bowlshaped_errors_to_rows_nl(plate_array, error)
    
    return plate_array



def __add_bowlshaped_errors_to_columns_nl(plate, error):
    num_rows, num_columns = plate.shape
    plate = np.asarray(plate, dtype=float)
    plate_array = plate.copy()
    
    translation_const = (num_columns - 1) / 2.0
    
    for row_index in range(num_rows):
        for col_index in range(num_columns):
            plate_array[row_index][col_index] = max(plate[row_index][col_index] * (1 + error*abs(col_index - translation_const)),0.0)
#            plate_array[row_index][col_index] += (np.random.random()-0.5)*error*abs(col_index - translation_const)
            
    return plate_array

def _add_row_linear_effect(plate, row_strengths, error):
    """Add error * row_strengths[row] to positive wells only."""
    plate = np.asarray(plate, dtype=float)
    plate_array = plate.copy()
    row_strengths = np.asarray(row_strengths, dtype=float)

    if row_strengths.shape != (plate_array.shape[0],):
        raise ValueError(
            "row_strengths must contain one value per plate row."
        )

    positive_wells = plate_array > 0
    offsets = error * row_strengths[:, np.newaxis]

    plate_array[positive_wells] += np.broadcast_to(
        offsets,
        plate_array.shape,
    )[positive_wells]

    return plate_array


def _multiply_row_effect(plate, row_strengths, error):
    """Multiply every well by 1 + error * row_strengths[row]."""
    plate = np.asarray(plate, dtype=float)
    plate_array = plate.copy()
    row_strengths = np.asarray(row_strengths, dtype=float)

    if row_strengths.shape != (plate_array.shape[0],):
        raise ValueError(
            "row_strengths must contain one value per plate row."
        )

    return plate_array * (
        1.0 + error * row_strengths[:, np.newaxis]
    )


def _add_column_linear_effect(plate, column_strengths, error):
    """Add error * column_strengths[column] to positive wells only."""
    plate = np.asarray(plate, dtype=float)
    plate_array = plate.copy()
    column_strengths = np.asarray(column_strengths, dtype=float)

    if column_strengths.shape != (plate_array.shape[1],):
        raise ValueError(
            "column_strengths must contain one value per plate column."
        )

    positive_wells = plate_array > 0
    offsets = error * column_strengths[np.newaxis, :]

    plate_array[positive_wells] += np.broadcast_to(
        offsets,
        plate_array.shape,
    )[positive_wells]

    return plate_array


def _multiply_column_effect(plate, column_strengths, error):
    """Multiply every well by 1 + error * column_strengths[column]."""
    plate = np.asarray(plate, dtype=float)
    plate_array = plate.copy()
    column_strengths = np.asarray(column_strengths, dtype=float)

    if column_strengths.shape != (plate_array.shape[1],):
        raise ValueError(
            "column_strengths must contain one value per plate column."
        )

    return plate_array * (
        1.0 + error * column_strengths[np.newaxis, :]
    )

def __add_bowlshaped_errors_to_rows_nl(plate, error):
    num_rows, num_columns = plate.shape
    plate = np.asarray(plate, dtype=float)
    plate_array = plate.copy()

    translation_const = (num_rows - 1) / 2.0
    
    for row_index in range(num_rows):
        for col_index in range(num_columns):
            plate_array[row_index][col_index] = max(plate[row_index][col_index] * (1 + error*abs(row_index - translation_const)),0.0)
#            plate_array[row_index][col_index] += (np.random.random()-0.5)*error*abs(col_index - translation_const)
            
    return plate_array


def add_errors_to_upper_rows(plate, error=0.125):
    num_rows, num_columns = plate.shape
    plate = np.asarray(plate, dtype=float)
    plate_array = plate.copy()
    
    for row_index in range(num_rows):
        for col_index in range(num_columns):
            plate_array[row_index][col_index] = plate[row_index][col_index] * (1 + error*(num_rows-1-row_index))
            
    return plate_array


def add_linear_errors_to_upper_rows(plate, error=8.0):
    """Add a top-to-bottom linear signal offset.

    The top row receives error * (num_rows - 1).
    The bottom row receives no added error.
    """
    num_rows, _ = plate.shape
    strengths = np.arange(num_rows - 1, -1, -1)

    return _add_row_linear_effect(plate, strengths, error)

def add_linear_errors_to_lower_rows(plate, error=8.0):
    """Add a bottom-directed linear signal offset.

    The top row receives no added error.
    The bottom row receives error * (num_rows - 1).
    """
    num_rows, _ = np.asarray(plate).shape
    strengths = np.arange(num_rows)

    return _add_row_linear_effect(plate, strengths, error)

def add_linear_errors_to_upper_rows_neg(plate, error=0.125):
    """Subtract a top-to-bottom linear offset without creating negatives.

    The top row receives the largest reduction.
    The bottom row is unchanged.
    """
    num_rows, num_columns = plate.shape
    plate = np.asarray(plate, dtype=float)
    plate_array = plate.copy()
    strengths = np.arange(num_rows - 1, -1, -1)
    offsets = error * strengths[:, np.newaxis]
    positive_wells = plate_array > 0

    reduced = np.maximum(plate_array - offsets, 0.0)

    return np.where(positive_wells, reduced, plate_array)

def add_linear_errors_to_lower_rows_neg(plate, error=0.125):
    """Subtract a bottom-directed linear offset without creating negatives.

    The top row is unchanged.
    The bottom row receives the largest reduction.
    """
    num_rows, num_columns = plate.shape
    plate = np.asarray(plate, dtype=float)
    plate_array = plate.copy()
    strengths = np.arange(num_rows)
    offsets = error * strengths[:, np.newaxis]
    positive_wells = plate_array > 0

    reduced = np.maximum(plate_array - offsets, 0.0)

    return np.where(positive_wells, reduced, plate_array)


def add_linear_errors_to_upper_rows_half(plate, error=0.125):
    """Add an upper-half linear signal offset.

    The row nearest the centre seam is unchanged. The top row receives
    the greatest additive offset.
    """
    num_rows, _ = np.asarray(plate).shape
    half_row = num_rows // 2
    strengths = np.zeros(num_rows, dtype=float)

    if half_row > 0:
        strengths[:half_row] = 2.0 * np.arange(
            half_row - 1,
            -1,
            -1,
        )

    return _add_row_linear_effect(plate, strengths, error)

def add_linear_errors_to_lower_rows_half(plate, error=0.125):
    """Add a lower-half linear signal offset.

    The row nearest the centre seam is unchanged. The bottom row receives
    the greatest additive offset.
    """
    num_rows, _ = np.asarray(plate).shape
    half_row = num_rows // 2
    strengths = np.zeros(num_rows, dtype=float)

    if num_rows > half_row:
        strengths[half_row:] = 2.0 * np.arange(
            num_rows - half_row
        )

    return _add_row_linear_effect(plate, strengths, error)

def add_linear_errors_to_left_columns_half(plate, error=0.125):
    """Add a left-half linear signal offset.

    The column nearest the centre seam is unchanged. The leftmost column
    receives the greatest additive offset.
    """
    _, num_columns = np.asarray(plate).shape
    half_column = num_columns // 2
    strengths = np.zeros(num_columns, dtype=float)

    if half_column > 0:
        strengths[:half_column] = 2.0 * np.arange(
            half_column - 1,
            -1,
            -1,
        )

    return _add_column_linear_effect(plate, strengths, error)

def add_linear_errors_to_right_columns_half(plate, error=0.125):
    """Add a right-half linear signal offset.

    The column nearest the centre seam is unchanged. The rightmost column
    receives the greatest additive offset.
    """
    _, num_columns = np.asarray(plate).shape
    half_column = num_columns // 2
    strengths = np.zeros(num_columns, dtype=float)

    if num_columns > half_column:
        strengths[half_column:] = 2.0 * np.arange(
            num_columns - half_column
        )

    return _add_column_linear_effect(plate, strengths, error)

def add_errors_to_lower_rows(plate, error=0.125):
    num_rows, num_columns = plate.shape
    plate = np.asarray(plate, dtype=float)
    plate_array = plate.copy()
    
    for row_index in range(num_rows):
        for col_index in range(num_columns):
            plate_array[row_index][col_index] = plate[row_index][col_index] * (1.0 + error*row_index)
            
    return plate_array


def add_errors_to_lower_rows_half(plate, error=0.125):
    """Apply a lower-half multiplicative gradient.

    The centre seam is unchanged. The bottom row is multiplied by
    1 + error.
    """
    num_rows, _ = np.asarray(plate).shape
    half_row = num_rows // 2
    strengths = np.zeros(num_rows, dtype=float)

    if num_rows > half_row:
        strengths[half_row:] = np.linspace(
            0.0,
            1.0,
            num_rows - half_row,
        )

    return _multiply_row_effect(plate, strengths, error)



def add_linear_errors_to_lower_rows_neg(plate, error=0.125):
    # No effect at the centre seam, maximum uplift at the bottom edge
    num_rows, num_columns = plate.shape
    plate = np.asarray(plate, dtype=float)
    plate_array = plate.copy()
    
    denominator = max(num_rows - 1 - half_row, 1)

    for row_index in range(half_row, num_rows):
        strength = (row_index - half_row) / denominator
        plate_array[row_index] *= 1.0 + error * strength
            
    return plate_array


def add_errors_to_left_columns(plate, error=0.125):
    num_rows, num_columns = plate.shape
    plate = np.asarray(plate, dtype=float)
    plate_array = plate.copy()
    
    for row_index in range(num_rows):
        for col_index in range(num_columns):
            plate_array[row_index][col_index] = plate[row_index][col_index] * (1 + error*(num_columns-1-col_index))
    return plate_array


def add_linear_errors_to_left_columns(plate, error=8.0):
    """Add a left-to-right linear signal offset.

    The leftmost column receives error * (num_columns - 1).
    The rightmost column receives no added error.
    """
    _, num_columns = np.asarray(plate).shape
    strengths = np.arange(num_columns - 1, -1, -1)

    return _add_column_linear_effect(plate, strengths, error)



def add_linear_errors_to_right_columns(plate, error=8.0):
    """Add a right-directed linear signal offset.

    The leftmost column receives no added error.
    The rightmost column receives error * (num_columns - 1).
    """
    _, num_columns = np.asarray(plate).shape
    strengths = np.arange(num_columns)

    return _add_column_linear_effect(plate, strengths, error)


def add_errors_to_left_columns_half(plate, error=0.125):
    """Apply a left-half multiplicative gradient.

    The centre seam is unchanged. The leftmost column is multiplied by
    1 + error.
    """
    _, num_columns = np.asarray(plate).shape
    half_column = num_columns // 2
    strengths = np.zeros(num_columns, dtype=float)

    if half_column > 0:
        strengths[:half_column] = np.linspace(
            1.0,
            0.0,
            half_column,
        )

    return _multiply_column_effect(plate, strengths, error)


def add_errors_to_right_columns_half(plate, error=0.125):
    """Apply a right-half multiplicative gradient.

    The centre seam is unchanged. The rightmost column is multiplied by
    1 + error.
    """
    _, num_columns = np.asarray(plate).shape
    half_column = num_columns // 2
    strengths = np.zeros(num_columns, dtype=float)

    if num_columns > half_column:
        strengths[half_column:] = np.linspace(
            0.0,
            1.0,
            num_columns - half_column,
        )

    return _multiply_column_effect(plate, strengths, error)



def add_striped_errors_even_rows_left(plate, error=0.125):
    num_rows, num_columns = plate.shape
    plate = np.asarray(plate, dtype=float)
    plate_array = plate.copy()
    
    for row_index in range(num_rows):
        if (row_index % 2 == 0):
            for col_index in range(num_columns):
                plate_array[row_index][col_index] = plate[row_index][col_index] * (1 + error*(num_columns-1-col_index))
                
    return plate_array



def add_striped_errors_odd_rows_left(plate, error=0.125):
    num_rows, num_columns = plate.shape
    plate = np.asarray(plate, dtype=float)
    plate_array = plate.copy()
    
    for row_index in range(num_rows):
        if (row_index % 2 == 1):
            for col_index in range(num_columns):
                plate_array[row_index][col_index] = plate[row_index][col_index] * (1 + error*(num_columns-1-col_index))
                
    return plate_array



def add_striped_errors_even_rows_right(plate, error=0.125):
    num_rows, num_columns = plate.shape
    plate = np.asarray(plate, dtype=float)
    plate_array = plate.copy()
    
    for row_index in range(num_rows):
        if (row_index % 2 == 0):
            for col_index in range(num_columns):
                plate_array[row_index][col_index] = plate[row_index][col_index] * (1 + error*(col_index))
                
    return plate_array




def add_striped_errors_odd_rows_right(plate, error=0.125):
    num_rows, num_columns = plate.shape
    plate = np.asarray(plate, dtype=float)
    plate_array = plate.copy()
    
    for row_index in range(num_rows):
        if (row_index % 2 == 1):
            for col_index in range(num_columns):
                plate_array[row_index][col_index] = plate[row_index][col_index] * (1 + error*(col_index))
                
    return plate_array

def add_striped_errors_pure_even(plate, error=0.10):
    """Pure interleaved-nozzle stripe disturbance.

    Models the systematic signal offset that arises when a multichannel
    liquid handler fills even and odd rows via separate nozzle banks with
    slightly different delivery volumes.  Even rows (row_index % 2 == 0)
    are scaled up by (1 + error); odd rows are left unchanged.  The effect
    is uniform across all columns — there is no within-row gradient — making
    this a clean periodic / structural row-bias disturbance that is
    qualitatively distinct from smooth spatial gradients.

    Physical calibration: inter-bank CV values of 2-5% for volume delivery
    translate to 5-20% signal differences in sensitive fluorescence assays
    (Mpindi et al., Bioinformatics 2015; liquid-handling CV literature).
    Recommended error levels: mild=0.05, moderate=0.10, strong=0.20.

    Parameters
    ----------
    plate : np.ndarray
        2-D array of well signal values.
    error : float
        Fractional signal uplift applied to even rows.  A value of 0.10
        means even-row wells are 10% higher than odd-row wells.

    Returns
    -------
    np.ndarray
        Disturbed plate array (same shape as input).
    """
    num_rows, num_columns = plate.shape
    plate = np.asarray(plate, dtype=float)
    plate_array = plate.copy()

    for row_index in range(num_rows):
        if row_index % 2 == 0:
            for col_index in range(num_columns):
                plate_array[row_index][col_index] = plate[row_index][col_index] * (1 + error)

    return plate_array


def lose_columns(plate, from_col, to_col, empty_value=0):
    num_rows, num_columns = plate.shape
    plate = np.asarray(plate, dtype=float)
    plate_array = plate.copy()
    
    from_col = int(np.clip(from_col, 0, num_columns))
    to_col = int(np.clip(to_col, 0, num_columns))
    
    if from_col > to_col:
        raise ValueError(
            "from_col must be less than or equal to to_col; "
            f"got {from_col} > {to_col}."
        )
    
    for col_index in range(from_col,to_col):
        for row_index in range(num_rows):
            plate_array[row_index][col_index] = empty_value
            
    return plate_array


def lose_rows(plate, from_row, to_row, empty_value=0):
    num_rows, num_columns = plate.shape
    plate = np.asarray(plate, dtype=float)
    plate_array = plate.copy()
    
    from_row = int(np.clip(from_row, 0, num_rows))
    to_row = int(np.clip(to_row, 0, num_rows))
    
    if from_row > to_row:
        raise ValueError(
            "from_row must be less than or equal to to_row; "
            f"got {from_row} > {to_row}."
        )
    
    for row_index in range(from_row,to_row):
        for col_index in range(num_columns):
            plate_array[row_index][col_index] = empty_value
            
    return plate_array


def add_exponential_errors_to_upper_rows(plate, error=0.125):
    num_rows, num_columns = plate.shape
    plate = np.asarray(plate, dtype=float)
    plate_array = plate.copy()
    
    for row_index in range(num_rows):
        for col_index in range(num_columns):
            plate_array[row_index][col_index] = plate[row_index][col_index] * ((1 + error)**(num_rows-1-row_index))
            
    return plate_array


def add_exponential_errors_to_lower_rows(plate, error=0.125):
    num_rows, num_columns = plate.shape
    plate = np.asarray(plate, dtype=float)
    plate_array = plate.copy()
    
    for row_index in range(num_rows):
        for col_index in range(num_columns):
            plate_array[row_index][col_index] = plate[row_index][col_index] * ((1 + error)**row_index)
            
    return plate_array



def add_exponential_errors_to_left_columns(plate, error=0.125):
    num_rows, num_columns = plate.shape
    plate = np.asarray(plate, dtype=float)
    plate_array = plate.copy()
    
    for row_index in range(num_rows):
        for col_index in range(num_columns):
            plate_array[row_index][col_index] = plate[row_index][col_index] * ((1 + error)**(num_columns-1-col_index))
            
    return plate_array



def add_exponential_errors_to_right_columns(plate, error=0.125):
    num_rows, num_columns = plate.shape
    plate = np.asarray(plate, dtype=float)
    plate_array = plate.copy()
    
    for row_index in range(num_rows):
        for col_index in range(num_columns):
            plate_array[row_index][col_index] = plate[row_index][col_index] * ((1 + error)**col_index)
            
    return plate_array


def add_linear_errors_to_bottom_left_half(plate, error=0.125):
    """Add additive linear lower-half and left-half disturbances."""
    plate_array = add_linear_errors_to_lower_rows_half(
        plate,
        error,
    )

    return add_linear_errors_to_left_columns_half(
        plate_array,
        error,
    )


def add_linear_errors_to_bottom_right_half(plate, error=0.125):
    """Add additive linear lower-half and right-half disturbances."""
    plate_array = add_linear_errors_to_lower_rows_half(
        plate,
        error,
    )

    return add_linear_errors_to_right_columns_half(
        plate_array,
        error,
    )


def add_linear_errors_to_upper_left(plate, error=0.125):
    plate_array = add_linear_errors_to_left_columns(plate, error)
    plate_array = add_linear_errors_to_upper_rows(plate_array, error)
    return plate_array



def add_diagonal_errors_x(plate, error=0.125):
    """
    Apply a top-left to bottom-right diagonal multiplicative gradient.
    Top-left is unchanged; bottom-right is multiplied by (1 + error).
    """
    num_rows, num_columns = plate.shape
    plate = np.asarray(plate, dtype=float)
    plate_array = plate.copy()
    
    row_denominator = max(num_rows - 1, 1)
        col_denominator = max(num_columns - 1, 1)

        for row_index in range(num_rows):
            for col_index in range(num_columns):
                row_strength = row_index / row_denominator
                col_strength = col_index / col_denominator
                diagonal_strength = (row_strength + col_strength) / 2.0

                plate_array[row_index, col_index] *= (
                    1.0 + error * diagonal_strength
                )
            
    return plate_array

def add_bottom_right_radial_errors(plate, error=0.125):
    """Apply a radial effect centred at the bottom-right corner.

    The bottom-right corner has the maximum multiplier (1 + error).
    The effect decreases linearly with distance and reaches zero at the
    most distant plate corner.
    """
    num_rows, num_columns = plate.shape
    plate = np.asarray(plate, dtype=float)
    plate_array = plate.copy()

    max_distance = math.hypot(
        max(num_rows - 1, 1),
        max(num_columns - 1, 1),
    )

    for row_index in range(num_rows):
        for col_index in range(num_columns):
            distance = math.hypot(
                num_rows - 1 - row_index,
                num_columns - 1 - col_index,
            )

            strength = 1.0 - distance / max_distance
            plate_array[row_index, col_index] *= (
                1.0 + error * max(strength, 0.0)
            )

    return plate_array


def add_diagonal_errors(plate, error=0.125):
    num_rows, num_columns = plate.shape
    plate = np.asarray(plate, dtype=float)
    plate_array = plate.copy()
    
    for row_index in range(num_rows):
        for col_index in range(num_columns):
            if math.sqrt((num_rows-row_index)**2 + (num_columns - col_index)**2) < num_columns:
                plate_array[row_index][col_index] = plate[row_index][col_index] * (1 + error*(num_columns - math.sqrt((num_rows-row_index)**2 + (num_columns - col_index)**2)))
#            plate_array[row_index][col_index] = plate[row_index][col_index] * (1 + error*(col_index - num_columns//2 + 1))
            
    return plate_array
