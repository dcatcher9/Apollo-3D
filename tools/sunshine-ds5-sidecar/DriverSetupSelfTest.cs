namespace Sunshine.Ds5Sidecar;

internal static class DriverSetupSelfTest
{
    internal static void Run()
    {
        foreach (var arguments in new[]
        {
            Array.Empty<string>(),
            new[] { "--probe" },
            new[] { "--install-drivers", "--install-drivers" },
            new[] { "--install-drivers", "--probe" },
            new[] { "--probe-host", "--install-drivers" },
            new[] { "--self-check", "--install-drivers" },
            new[] { "--install-drivers", "--self-test", "standard" },
            new[] { "--pipe", "test-pipe", "--install-drivers" },
            new[] { "--install-drivers", "--enable-composite-microphone-prototype" },
            new[] { "--install-drivers", "--result", "result.json" },
        })
        {
            using var backend = new FakeBackend();
            var result = DriverSetup.Execute(arguments, backend);
            Require(!result.Success && result.ExitCode != 0 && result.Error == "invalid_arguments" &&
                    backend.Reads == 0 && backend.Installs.Count == 0,
                "mixed installation mode is rejected before platform access");
        }

        foreach (var version in new string?[] { null, "bad", "1.6.2", "1.6.1.0", "1.6.3.0", "2.0.0.0" })
        {
            using var backend = new FakeBackend { Version = version };
            var result = Install(backend);
            Require(result.Error == "unsupported_runtime" && backend.Installs.Count == 0,
                "only the exact pinned Core version can install drivers");
        }
        using (var backend = new FakeBackend { Elevated = false })
        {
            var result = Install(backend);
            Require(result.Error == "administrator_required" && backend.Installs.Count == 0,
                "non-administrator setup cannot install drivers");
        }
        using (var backend = new FakeBackend { SunshineRunning = true })
        {
            var result = Install(backend);
            Require(result.Error == "sunshine_running" && backend.Installs.Count == 0,
                "a running Sunshine host prevents driver setup");
        }

        foreach (var hidReady in new[] { false, true })
        foreach (var usbipReady in new[] { false, true })
        {
            using var backend = new FakeBackend { HidReady = hidReady, UsbipReady = usbipReady };
            var result = Install(backend);
            var expected = new List<string>();
            if (!hidReady) expected.Add("hid");
            if (!usbipReady) expected.Add("usbip");
            Require(result.Success && result.ExitCode == 0 && result.DriverInstalled == true &&
                    result.UsbipAvailable == true && result.HidInstallAttempted == !hidReady &&
                    result.UsbipInstallAttempted == !usbipReady && backend.Installs.SequenceEqual(expected),
                "setup installs only missing backends, preserving installed controllers");
        }
        using (var backend = new FakeBackend { FailHid = true })
        {
            var result = Install(backend);
            Require(!result.Success && result.Error == "setup_failed" && result.Message == "HID install error" &&
                    result.HidInstallAttempted && !result.UsbipInstallAttempted &&
                    backend.Installs.SequenceEqual(new[] { "hid" }),
                "a HID installation exception prevents later USB/IP installation");
        }
        using (var backend = new FakeBackend { FailUsbip = true })
        {
            var result = Install(backend);
            Require(!result.Success && result.ExitCode != 0 && result.Error == "setup_failed" &&
                    result.Message == "USB/IP install error" && result.DriverInstalled == true &&
                    result.UsbipAvailable == false && result.UsbipInstallAttempted,
                "USB/IP failure reports partial readiness and nonzero exit status");
        }
        using (var backend = new FakeBackend { HidInstallHasNoEffect = true })
        {
            var result = Install(backend);
            Require(!result.Success && result.Error == "hid_not_ready" && !result.UsbipInstallAttempted,
                "HID success requires the installed-driver postcondition");
        }
        using (var backend = new FakeBackend { UsbipInstallHasNoEffect = true })
        {
            var result = Install(backend);
            Require(!result.Success && result.Error == "drivers_not_ready" && result.UsbipAvailable == false,
                "USB/IP success requires a usable transport, not just an installer return");
        }
        using (var backend = new FakeBackend { StartSunshineAfterHid = true })
        {
            var result = Install(backend);
            Require(!result.Success && result.Error == "sunshine_running" && result.DriverInstalled == true &&
                    !result.UsbipInstallAttempted,
                "host start during setup prevents the next driver mutation");
        }
    }

    private static DriverSetup.Result Install(FakeBackend backend) =>
        DriverSetup.Execute(new[] { "--install-drivers" }, backend);

    private static void Require(bool value, string contract)
    {
        if (!value) throw new InvalidOperationException("Driver setup contract: " + contract);
    }

    private sealed class FakeBackend : IDriverSetupBackend
    {
        public string? Version = "1.6.2.0";
        public bool Elevated = true;
        public bool SunshineRunning;
        public bool HidReady;
        public bool UsbipReady;
        public bool FailHid;
        public bool FailUsbip;
        public bool HidInstallHasNoEffect;
        public bool UsbipInstallHasNoEffect;
        public bool StartSunshineAfterHid;
        public int Reads;
        public List<string> Installs { get; } = new();
        public string? RuntimeVersion { get { Reads++; return Version; } }
        public bool IsElevated { get { Reads++; return Elevated; } }
        public bool IsSunshineRunning { get { Reads++; return SunshineRunning; } }
        public bool IsDriverInstalled { get { Reads++; return HidReady; } }
        public bool IsUsbipAvailable { get { Reads++; return UsbipReady; } }
        public void InstallHidDriver()
        {
            Installs.Add("hid");
            if (FailHid) throw new InvalidOperationException("HID install error");
            if (!HidInstallHasNoEffect) HidReady = true;
            if (StartSunshineAfterHid) SunshineRunning = true;
        }
        public void InstallUsbipBackend()
        {
            Installs.Add("usbip");
            if (FailUsbip) throw new InvalidOperationException("USB/IP install error");
            if (!UsbipInstallHasNoEffect) UsbipReady = true;
        }
        public void Dispose() { }
    }
}
