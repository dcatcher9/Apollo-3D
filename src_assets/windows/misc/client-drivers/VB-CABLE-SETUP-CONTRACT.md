# Pinned VB-CABLE setup contract

The optional installer uses the original [Pack45 archive](https://download.vb-audio.com/Download_CABLE/VBCABLE_Driver_Pack45.zip)
from VB-Audio. Its supplied readme requires elevation, all extracted package files,
and a Windows restart after installing. The readme does not describe command-line
switches. The unattended switches below were verified by read-only static
inspection of this exact signed executable on 2026-09-09; the installer was not run.

- Archive SHA-256: `b950e39f01af1d04ea623c8f6d8eb9b6ea5c477c637295fabf20631c85116bfb`
- `VBCABLE_Setup_x64.exe` SHA-256: `734c35dfa6d98f48782a451633ceb471166ec70d60482fd89a1123d0ee3c4f41`
- Valid Authenticode signer: `BUREL VINCENT Entrepreneur individuel`
- Signer thumbprint: `A77952D93229D0EC36E2543081EEA7D125732B9C`

Addresses below are relative virtual addresses (RVAs), with image base
`0x140000000`. They can be reproduced with `objdump -d -Mintel` on the unmodified
executable, without executing it.

| Location | Verified behavior |
| --- | --- |
| `0x2d20`–`0x2dc5` | The command parser scans `-` switches. Lowercase `i` (`0x69`) sets install flag `0x1bb58`; `h`/`H` sets hidden flag `0x1b744`; `u`/`U` sets a separate uninstall flag `0x1bb5c`. |
| `0x6642`–`0x664f` | WinMain passes its command-line argument to that parser. |
| `0x62d9`–`0x6309` | An install or uninstall flag posts the operation command automatically. |
| `0x6448`–`0x647f` | For an absent device, `-i` calls the install routine at `0x5270`. For an existing device, removal requires the separate `-u` flag; `-i` alone does not remove it. The operation's return value is recorded at `0x1c038`. |
| `0x65d5`–`0x65f8` | `-h` selects `ShowWindow(..., SW_HIDE)` instead of showing the setup window. |
| `0x535c`–`0x5386`, `0x5d3d`–`0x5e34` | `-h` skips the install error dialog and the successful-install dialog/browser launch. |
| `0x5e34`, `0x6753` | Successful install returns zero. WinMain returns the recorded operation status. Zero also occurs for a no-op, so exit code alone is insufficient evidence of installation. |

The PowerShell wrapper passes exactly `-i -h` in silent mode, never `-u`, and
requires the signed INF's actual `VBAudioVACWDM` MEDIA-class device with no problem
or a restart-required status after zero exit. It maps a verified new installation
to `3010` because the vendor requires a restart. An already healthy device maps to
`0`; an already pending restart maps to `3010`; disabled/problem devices are left
alone with failure. Nonzero vendor exits, timeout, cancellation, or absent devices
map to failure. No default endpoint, certificate trust, or host configuration is
changed. A timed-out driver installer is not killed partway through an operation.

Package, log, staging, and executable paths must be local absolute paths without
reparse ancestors. Owners and write access are limited to SYSTEM, Administrators,
and TrustedInstaller; ancestors must also prevent unprivileged replacement of the
protected subtree. Fresh staging directories are created atomically with a
protected SYSTEM/Administrators-only DACL. A user-writable custom installation or
log location fails before extraction or execution; the installer does not repair
its ACL or import publisher certificates to weaken Windows driver trust.

Changing the package or setup hash requires re-verifying this contract; these
observations do not assert a stable CLI for other VB-Audio products or versions.
