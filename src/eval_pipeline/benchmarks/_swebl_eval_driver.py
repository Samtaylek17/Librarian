"""Driver for the vendored SWE-bench-Live harness with a remote-docker-safe
patch path.

The upstream harness (``launch.core.runtime.SetupRuntime.apply_patch``)
writes the diff to a host directory that is bind-mounted into the eval
container.  With ``DOCKER_HOST=ssh://<host>`` the bind source resolves on
the *remote* daemon's filesystem, so the diff written locally is invisible
in the container and ``git apply`` fails with "can't open patch" — while
the harness still reports a result from the unpatched repo.

This driver monkey-patches ``apply_patch`` to ship the diff through the
docker API (``put_archive``), which works for local and remote daemons
alike, then runs ``evaluation.evaluation`` unchanged.  The vendored tree
stays byte-identical (vendoring policy: adapters only).

Usage: invoked by SWEBenchLiveEvaluator with cwd = vendored harness root;
all CLI args are forwarded verbatim to ``evaluation.evaluation``.
"""

import io
import os
import runpy
import sys
import tarfile
import uuid

# cwd is the vendored harness root; mirror `python -m`'s sys.path[0].
sys.path.insert(0, os.getcwd())

from launch.core import runtime as _rt  # noqa: E402


def _apply_patch_via_api(self, patch: str, verbose=False) -> bool:
    output_temp = "\n\n<<<<<<PATCH FAILED TO APPLY CLEANLY\n{out}\n>>>>>>\n\n"

    filename = f"{uuid.uuid4()}.diff"
    data = patch.encode("utf-8")
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w") as tar:
        info = tarfile.TarInfo(name=filename)
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))
    stream.seek(0)

    self.send_command(f"mkdir -p {self.mnt_container}")
    self.container.put_archive(self.mnt_container, stream)

    containerpath = os.path.join(self.mnt_container, filename)
    cmd = f"""git apply --reject  --whitespace=nowarn  {containerpath} """
    res = self.send_command(cmd)
    self.send_command(f"rm {containerpath}")
    if int(res.metadata.exit_code) == 0:
        print(f"{cmd} ---- Patch applied Successfully!", flush=True)
        return True
    else:
        if verbose:
            print(output_temp.format(out=res.output), flush=True)
        return False


_rt.SetupRuntime.apply_patch = _apply_patch_via_api

sys.argv = ["evaluation.evaluation"] + sys.argv[1:]
runpy.run_module("evaluation.evaluation", run_name="__main__")
