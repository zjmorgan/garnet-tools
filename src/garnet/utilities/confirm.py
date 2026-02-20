import os
import sys
import subprocess

CONFIRM_DATA_BIN = "/SNS/software/nses/bin/confirm-data"


def _extract_run_id(log_filename):
    """
    Extract run identifier from reduction log filename.

    Parameters
    ----------
    log_filename : str
        Name of a reduction log file. Expected format resembles
        ``<prefix>_<run>.<ext>``.

    Returns
    -------
    str or None
        Parsed run identifier if extraction succeeds, otherwise None.
    """
    base = os.path.basename(log_filename)

    if "_" not in base:
        return None

    run_part = base.split("_", 1)[1].split(".", 1)[0]
    return run_part if run_part else None


def _scan_reduction_logs(directory):
    """
    Scan reduction_log directory for run status.

    Parameters
    ----------
    directory : str
        Path to the reduction_log directory.

    Returns
    -------
    set
        Runs with associated ``.log`` files.
    set
        Runs with non-empty ``.err`` files.
    """
    all_runs = set()
    failing_runs = set()

    for name in os.listdir(directory):
        path = os.path.join(directory, name)

        if name.endswith(".log"):
            run_id = _extract_run_id(name)
            if run_id is not None:
                all_runs.add(run_id)

        elif name.endswith(".err"):
            try:
                if os.stat(path).st_size != 0:
                    run_id = _extract_run_id(name)
                    if run_id is not None:
                        failing_runs.add(run_id)
            except OSError:
                pass

    return all_runs, failing_runs


def _confirm_data(instrument, ipts, status):
    """
    Execute confirm-data command.

    Parameters
    ----------
    instrument : str
        Instrument name.

    ipts : str
        IPTS identifier string (e.g. ``IPTS-1234``).

    status : str
        Confirmation status (``Yes``, ``No``, ``Partially``, ``Unknown``).

    Returns
    -------
    subprocess.CompletedProcess
        Result of subprocess execution.
    """
    ipts_number = ipts.split("-", 1)[-1]

    cmd = [
        CONFIRM_DATA_BIN,
        instrument,
        ipts_number,
        "1",
        "Auto",
        "-s",
        status,
    ]

    return subprocess.run(cmd, check=False)


def confirm_autoreduce(filename):
    """
    Confirm autoreduction status based on reduction logs.

    Parameters
    ----------
    filename : str
        Full data path containing facility, instrument, and IPTS fields.

    Notes
    -----
    Confirmation logic:

    * ``Unknown``   → No ``.log`` files detected
    * ``Yes``       → Logs exist and no failures detected
    * ``Partially`` → Some runs failed
    * ``No``        → All runs failed
    """
    parts = filename.split("/")

    if len(parts) < 4:
        print("[Confirm data error] Unexpected filename format.")
        print("[Confirm data error] Skipping.")
        return

    facility, instrument, ipts = parts[1:4]

    directory = os.path.join(
        "/",
        facility,
        instrument,
        ipts,
        "shared",
        "autoreduce",
        "reduction_log",
    )

    if not os.path.isdir(directory):
        print("[Confirm data error] Reduction_log directory not found.")
        print("[Confirm data error] Skipping.")
        print("[Confirm data error]", directory)
        return

    all_runs, failing_runs = _scan_reduction_logs(directory)

    if not all_runs:
        status = "Unknown"
        print("[Confirm data info] No valid .log files found.")
        print("[Confirm data info] Thus confirmed 'Unknown'.")

    elif not failing_runs:
        status = "Yes"
        print("[Confirm data info] All data processed successfully.")
        print("[Confirm data info] Thus confirmed 'Yes'.")

    elif failing_runs < all_runs:
        status = "Partially"
        print("[Confirm data info] Failures encountered.")
        print("[Confirm data info] Thus confirmed 'Partially'.")
        print("[Confirm data info] Failing runs:", sorted(failing_runs))

    else:
        status = "No"
        print("[Confirm data info] All data processing failed.")
        print("[Confirm data info] Thus confirmed 'No'.")
        print("[Confirm data info] Failing runs:", sorted(failing_runs))

    confirm_output = _confirm_data(instrument, ipts, status)

    print("[Confirm data info] Data confirmation output:")
    print(confirm_output)


if __name__ == "__main__":
    filename = sys.argv[1]
    confirm_autoreduce(filename)
