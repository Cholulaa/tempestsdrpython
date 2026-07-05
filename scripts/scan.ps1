<#
.SYNOPSIS
  Sweep a frequency range with an RTL-SDR and rank tunings by how strongly a
  video frame-rate shows up — a calibration helper to find where a target
  display leaks.

.DESCRIPTION
  For each frequency it records a short capture with rtl_sdr, runs
  `tempestsdr detect --json`, and collects the frame-rate confidence
  (autocorrelation peak prominence). Candidates with confidence well above ~2
  and a plausible refresh rate (55-90 Hz) are worth reconstructing.

  Run from an environment where `rtl_sdr` and `python -m tempestsdr.cli` both
  work (e.g. an activated radioconda prompt).

.EXAMPLE
  .\scan.ps1 -Start 250e6 -Stop 600e6 -Step 25e6

.EXAMPLE
  .\scan.ps1 -Start 300e6 -Stop 500e6 -Step 10e6 -Seconds 6 -SampleRate 2.4e6
#>
param(
  [double]$Start      = 300e6,      # sweep start (Hz)
  [double]$Stop       = 600e6,      # sweep stop  (Hz)
  [double]$Step       = 25e6,       # sweep step  (Hz)
  [double]$SampleRate = 2.4e6,      # RTL-SDR sample rate (<= 2.4 Msps is safe)
  [double]$Seconds    = 4,          # capture length per frequency
  [string]$RtlSdr     = "rtl_sdr",  # path to rtl_sdr(.exe)
  [string]$TmpFile    = "scan_tmp.iq"
)

$n = [long]($SampleRate * $Seconds)
$results = @()

Write-Host "Sweeping $($Start/1e6) - $($Stop/1e6) MHz, step $($Step/1e6) MHz, $Seconds s each`n"

for ($f = $Start; $f -le $Stop; $f += $Step) {
    $mhz = "{0,8:N3}" -f ($f / 1e6)
    Write-Host "  $mhz MHz ..." -NoNewline

    & $RtlSdr -f $f -s $SampleRate -n $n $TmpFile 2>$null
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path $TmpFile)) {
        Write-Host " capture failed"; continue
    }

    $raw = & python -m tempestsdr.cli detect $TmpFile --samplerate $SampleRate --format uint8 --json
    try   { $j = $raw | ConvertFrom-Json } catch { Write-Host " detect error"; continue }

    if ($j.ok) {
        $flag = if ($j.confidence -gt 2 -and $j.refresh_rate -ge 55 -and $j.refresh_rate -le 90) { "  <== candidate" } else { "" }
        Write-Host ("  refresh={0,6:N2} Hz  conf={1,7:N1}{2}" -f $j.refresh_rate, $j.confidence, $flag)
        $results += [pscustomobject]@{
            FreqMHz    = [double]("{0:N3}" -f ($f / 1e6))
            RefreshHz  = [double]("{0:N2}" -f $j.refresh_rate)
            Confidence = [double]("{0:N1}" -f $j.confidence)
            Lines      = $j.height_lines
            Preset     = $j.closest_preset
        }
    } else {
        Write-Host " no lock"
    }
}

if (Test-Path $TmpFile) { Remove-Item $TmpFile -ErrorAction SilentlyContinue }

Write-Host "`n=== Best tunings (ranked by confidence) ==="
$results | Sort-Object Confidence -Descending | Select-Object -First 12 | Format-Table -AutoSize

Write-Host "Next: pick the highest-confidence frequency with a 55-90 Hz refresh, then:"
Write-Host "  rtl_sdr -f <FREQ> -s $SampleRate -n $([long]($SampleRate*10)) capture.iq"
Write-Host "  python -m tempestsdr.cli reconstruct capture.iq out.png --samplerate $SampleRate --format uint8 --mode `"<PRESET>`" --motion-blur 0.7"
