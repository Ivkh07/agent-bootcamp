"""Code interpreter tool."""

import os
from pathlib import Path
from typing import Sequence
import base64
from e2b_code_interpreter import AsyncSandbox
from pydantic import BaseModel

from ..async_utils import gather_with_progress
import json

class _CodeInterpreterOutputError(BaseModel):
    """Error from code interpreter."""

    name: str
    value: str
    traceback: str
    
def save_to_db_or_file(result: dict):
        """Custom persistence hook: Save to DB or file."""
        # You can adjust this to save to a DB or a file depending on your needs.
        # In this case, we'll save to a file for demonstration purposes.
        save_to_file(result)

def save_to_file(result: dict):
        """Save the execution result to a JSON file."""
        log_file = "execution_logs.json"
        with open(log_file, "a") as f:
            json.dump(result, f)
            f.write("\n")  # New line to separate entries for easier reading

class CodeInterpreterOutput(BaseModel):
    """Output from code interpreter."""

    stdout: list[str]
    stderr: list[str]
    error: _CodeInterpreterOutputError | None = None
    downloaded_local_path: str | None = None  # New field
    download_success: bool = False             # New field

    def __init__(self, stdout: list[str], stderr: list[str], **kwargs):
        """Split lines in stdout and stderr."""
        stdout_processed = []
        for _line in stdout:
            stdout_processed.extend(_line.splitlines())

        stderr_processed = []
        for _line in stderr:
            stderr_processed.extend(_line.splitlines())

        super().__init__(stdout=stdout_processed, stderr=stderr_processed, **kwargs)


async def _upload_file(sandbox: "AsyncSandbox", local_path: "str | Path") -> str:
    """Upload file to sandbox.

    Returns
    -------
        str, denoting the remote path.
    """
    path = Path(local_path)
    remote_path = f"{path.name}"
    with open(local_path, "rb") as file:
        await sandbox.files.write(remote_path, file)

    return remote_path


async def _upload_files(
    sandbox: "AsyncSandbox", paths: Sequence[Path | str]
) -> list[str]:
    """Upload files to the sandbox.

    Parameters
    ----------
        paths: Sequence[pathlib.Path | str]
            Files to upload to the sandbox.

    Returns
    -------
        list[str]
        List of remote paths, one per file.
    """
    if not paths:
        return []

    file_upload_coros = [_upload_file(sandbox, _path) for _path in paths]
    remote_paths = await gather_with_progress(
        file_upload_coros, description=f"Uploading {len(paths)} to sandbox"
    )
    return list(remote_paths)


def _enumerate_files(base_path: str | Path) -> list[Path]:
    """
    Recursively enumerate all files under a directory.

    Args
    ----
        base_path: Path to the starting directory.
            If input is a file, that file alone will be returned.

    Returns
    -------
        list[str]: List of file paths.
    """
    if os.path.isfile(base_path):
        return [Path(base_path)]

    file_list = []
    for root, _, files in os.walk(base_path):
        for name in files:
            file_list.append(Path(root) / name)
    return file_list


class CodeInterpreter:
    """Code Interpreter tool for the agent."""

    def __init__(
        self,
        local_files: "Sequence[Path | str]| None" = None,
        timeout_seconds: int = 30,
        template_name: str | None = None,
        sandbox_output_file: str | None = None,
        local_save_directory: str = "local_sandbox_downloads" # Default to a safe directory
    ):
        """Configure your Code Interpreter session.

        Note that the sandbox is not persistent, and each run_code will
        execute in a fresh sandbox! (e.g., variables need to be re-declared each time.)

        Parameters
        ----------
            local_files : list[pathlib.Path | str] | None
                Optionally, specify a list of local files (as paths)
                to upload to sandbox working directory. Folders will be flattened.
            timeout_seconds : int
                Limit executions to this duration.
            template_name : str | None
                Optionally, override the default e2b template name.
                See e2b_template.md for details.
        """
        self.timeout_seconds = timeout_seconds
        self.local_files = []
        self.template_name = template_name

        # Recursively find files if the given path is a folder.
        if local_files:
            for _path in local_files:
                self.local_files.extend(_enumerate_files(_path))
        self.template_name = template_name

    async def run_code(self, code: str, 
                       sandbox_output_file: str | None = None,
        local_save_directory: str = "local_sandbox_downloads" # Default to a safe directory
        ) -> str:
        """Run the given Python code in a sandbox environment.

        Parameters
        ----------
            code : str
                Python logic to execute.
        """
        sbx = await AsyncSandbox.create(
            timeout=self.timeout_seconds, template=self.template_name
        )
        await _upload_files(sbx, self.local_files)

        try:
            result = await sbx.run_code(
                code, on_error=lambda error: print(error.traceback)
            )
            response = CodeInterpreterOutput.model_validate_json(result.logs.to_json())


            fname = 'result.csv'
            try:
            
                file_content_bytes = await sbx.files.read (fname)
                local_dir = Path(local_save_directory)
                local_dir.mkdir(parents=True, exist_ok=True) # Ensure local directory exists

                local_path = local_dir / Path(fname)


                with open(local_path, "w") as f:
                    f.write(file_content_bytes)

                response.downloaded_local_path = str(local_path)
                response.download_success = True
                print(f"Successfully downloaded '{fname}' from sandbox to '{local_path}'")
            except Exception as e:
                print(f"Error downloading '{fname}' from sandbox: {e}")
                response.download_success = False # Indicate failure

            error = result.error
            if error is not None:
                response.error = _CodeInterpreterOutputError.model_validate_json(
                    error.to_json()
                )
            execution_result  = {
                "code": code,
                "results": result.logs.to_json(),
                "stdout": result.logs.stdout,}

            save_to_db_or_file(execution_result)
            return response.model_dump_json()
        finally:
            await sbx.kill()
