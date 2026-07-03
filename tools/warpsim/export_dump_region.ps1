# Exports the raw binary inputs the warpsim tools consume from a "Dump 3D" folder.
#
#   .\export_dump_region.ps1 -Dump E:\ApolloDev\sbs_dump\dump_20260703_011224_03 `
#                            -SrcX 1900 -SrcY 100 -SrcW 800 -SrcH 800 [-OutDir .]
#
# Produces (in -OutDir):
#   depth_<DW>x<DH>.bin                    full guided depth, R-channel bytes
#   source_x<X>y<Y>_<W>x<H>.bin            RGB24 source crop (include >=40px margins:
#                                          the warp's search radius is ~35 src px)
#   gt_left_...bin / gt_right_...bin       RGB24 ground-truth crops of both SBS eyes
#                                          at the matching region (eye = 0.8 * source)
# and prints the region constants to paste into a warpsim_*.cpp copy.
param(
    [Parameter(Mandatory)][string]$Dump,
    [Parameter(Mandatory)][int]$SrcX,
    [Parameter(Mandatory)][int]$SrcY,
    [Parameter(Mandatory)][int]$SrcW,
    [Parameter(Mandatory)][int]$SrcH,
    [string]$OutDir = "."
)
Add-Type -AssemblyName System.Drawing

function Export-Raw($bmpPath, $rx, $ry, $rw, $rh, $outPath, $depthMode) {
    $bmp = [System.Drawing.Bitmap]::FromFile($bmpPath)
    try {
        if ($rw -eq 0) { $rw = $bmp.Width; $rh = $bmp.Height; $rx = 0; $ry = 0 }
        $rect = New-Object System.Drawing.Rectangle($rx, $ry, $rw, $rh)
        $crop = $bmp.Clone($rect, [System.Drawing.Imaging.PixelFormat]::Format32bppArgb)
        try {
            $data = $crop.LockBits((New-Object System.Drawing.Rectangle(0, 0, $rw, $rh)),
                [System.Drawing.Imaging.ImageLockMode]::ReadOnly,
                [System.Drawing.Imaging.PixelFormat]::Format32bppArgb)
            $bytes = New-Object byte[] ($data.Stride * $rh)
            [System.Runtime.InteropServices.Marshal]::Copy($data.Scan0, $bytes, 0, $bytes.Length)
            $crop.UnlockBits($data)
            if ($depthMode) {
                $out = New-Object byte[] ($rw * $rh)
                for ($y = 0; $y -lt $rh; $y++) {
                    for ($x = 0; $x -lt $rw; $x++) { $out[$y * $rw + $x] = $bytes[$y * $data.Stride + $x * 4 + 2] }
                }
            } else {
                $out = New-Object byte[] ($rw * $rh * 3)
                for ($y = 0; $y -lt $rh; $y++) {
                    for ($x = 0; $x -lt $rw; $x++) {
                        $si = $y * $data.Stride + $x * 4; $di = ($y * $rw + $x) * 3
                        $out[$di] = $bytes[$si + 2]; $out[$di + 1] = $bytes[$si + 1]; $out[$di + 2] = $bytes[$si]
                    }
                }
            }
            [IO.File]::WriteAllBytes($outPath, $out)
            Write-Host "wrote $outPath ($rw x $rh)"
            return @($rw, $rh)
        } finally { if ($crop -ne $bmp) { $crop.Dispose() } }
    } finally { $bmp.Dispose() }
}

$src = [System.Drawing.Bitmap]::FromFile("$Dump\source.png"); $fw = $src.Width; $fh = $src.Height; $src.Dispose()
$sbs = [System.Drawing.Bitmap]::FromFile("$Dump\sbs.png"); $sw2 = $sbs.Width; $sh2 = $sbs.Height; $sbs.Dispose()
$eyeW = [int]($sw2 / 2); $scale = $eyeW / $fw   # eye coords = scale * source coords

$dd = Export-Raw "$Dump\depth.png" 0 0 0 0 "$OutDir\depth_full.bin" $true
Move-Item "$OutDir\depth_full.bin" "$OutDir\depth_$($dd[0])x$($dd[1]).bin" -Force
Export-Raw "$Dump\source.png" $SrcX $SrcY $SrcW $SrcH "$OutDir\source_x${SrcX}y${SrcY}_${SrcW}x${SrcH}.bin" $false | Out-Null

$ex = [int]($SrcX * $scale); $ey = [int]($SrcY * $scale)
$ew = [int]($SrcW * $scale); $eh = [int]($SrcH * $scale)
Export-Raw "$Dump\sbs.png" $ex $ey $ew $eh "$OutDir\gt_left_x${ex}y${ey}_${ew}x${eh}.bin" $false | Out-Null
Export-Raw "$Dump\sbs.png" ($ex + $eyeW) $ey $ew $eh "$OutDir\gt_right_x$($ex+$eyeW)y${ey}_${ew}x${eh}.bin" $false | Out-Null

Write-Output ""
Write-Output "--- paste into your warpsim_*.cpp copy ---"
Write-Output "static const int DW = $($dd[0]), DH = $($dd[1]);"
Write-Output "static const int SW = $SrcW, SH = $SrcH;"
Write-Output "static const int SOX = $SrcX, SOY = $SrcY;"
Write-Output "static const int OW = $ew, OH = $eh;          // or a sub-region of it"
Write-Output "static const int EOX = $ex, EOY = $ey;"
Write-Output "// source frame: ${fw}x${fh}; sbs: ${sw2}x${sh2}; eye scale: $scale"
