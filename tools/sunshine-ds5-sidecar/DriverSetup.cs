using System.Diagnostics;
using System.Reflection;
using System.Security.Principal;
using System.Text.Json;
using HIDMaestro;

namespace Sunshine.Ds5Sidecar;

internal interface IDriverSetupBackend : IDisposable
{
    string? RuntimeVersion { get; }
    bool IsElevated { get; }
    bool IsSunshineRunning { get; }
    bool IsDriverInstalled { get; }
    bool IsUsbipAvailable { get; }
    void InstallHidDriver();
    void InstallUsbipBackend();
}

/// <summary>
/// Explicit installer entry point. Streaming and readiness checks never call
/// this path, and setup does not create controllers or submit demo input.
/// </summary>
internal static class DriverSetup
{
    internal sealed class Result
    {
        public string Operation => "install-drivers";
        public bool Success { get; set; }
        public string? RuntimeVersion { get; set; }
        public bool? Elevated { get; set; }
        public bool? DriverInstalled { get; set; }
        public bool? UsbipAvailable { get; set; }
        public bool HidInstallAttempted { get; set; }
        public bool UsbipInstallAttempted { get; set; }
        public string? Error { get; set; }
        public string? Message { get; set; }
        public int ExitCode => Success ? 0 : 1;
    }

    internal static int Run(string[] arguments)
    {
        var output = Console.Out;
        // HIDMaestro's install routines also write progress to Console.Out.
        // Keep stdout a single JSON result for the installer bootstrap.
        Console.SetOut(Console.Error);
        Result result;
        try
        {
            using var backend = new WindowsBackend();
            result = Execute(arguments, backend);
        }
        catch (Exception error)
        {
            result = Fail(new Result(), "setup_failed", error.Message);
        }
        output.WriteLine(JsonSerializer.Serialize(result, new JsonSerializerOptions
        {
            PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower,
        }));
        return result.ExitCode;
    }

    internal static Result Execute(string[] arguments, IDriverSetupBackend backend)
    {
        var result = new Result();
        if (arguments.Length != 1 || arguments[0] != "--install-drivers")
            return Fail(result, "invalid_arguments",
                "--install-drivers must be used alone, without stream, probe, or test options.");

        var mayReadReadiness = false;
        try
        {
            result.RuntimeVersion = backend.RuntimeVersion;
            if (!Version.TryParse(result.RuntimeVersion, out var version) || version != new Version(1, 6, 2, 0))
                return Fail(result, "unsupported_runtime", "Driver setup requires HIDMaestro Core 1.6.2.0.");
            result.Elevated = backend.IsElevated;
            if (result.Elevated != true)
                return Fail(result, "administrator_required", "Run driver setup as Administrator.");
            if (backend.IsSunshineRunning)
                return Fail(result, "sunshine_running", "Stop Sunshine before installing optional drivers.");

            mayReadReadiness = true;
            result.DriverInstalled = backend.IsDriverInstalled;
            result.UsbipAvailable = backend.IsUsbipAvailable;
            if (result.DriverInstalled != true)
            {
                if (backend.IsSunshineRunning)
                    return Fail(result, "sunshine_running", "Sunshine started during driver setup; stop it and retry.");
                // InstallDriver sweeps virtual devices even on its same-version
                // fast path. Never call it merely to check an installed driver.
                result.HidInstallAttempted = true;
                backend.InstallHidDriver();
                result.DriverInstalled = backend.IsDriverInstalled;
                if (result.DriverInstalled != true)
                    return Fail(result, "hid_not_ready", "The HIDMaestro driver is still unavailable after setup.");
            }
            if (result.UsbipAvailable != true)
            {
                if (backend.IsSunshineRunning)
                    return Fail(result, "sunshine_running", "Sunshine started during driver setup; stop it and retry.");
                result.UsbipInstallAttempted = true;
                backend.InstallUsbipBackend();
            }

            result.DriverInstalled = backend.IsDriverInstalled;
            result.UsbipAvailable = backend.IsUsbipAvailable;
            if (result.DriverInstalled != true || result.UsbipAvailable != true)
                return Fail(result, "drivers_not_ready", "Optional controller drivers are not ready after setup.");
            result.Success = true;
            return result;
        }
        catch (Exception error)
        {
            // Report a partial installation accurately without replacing the
            // original error if a readiness probe also fails.
            if (mayReadReadiness)
            {
                try { result.DriverInstalled = backend.IsDriverInstalled; } catch { }
                try { result.UsbipAvailable = backend.IsUsbipAvailable; } catch { }
            }
            return Fail(result, "setup_failed", error.Message);
        }
    }

    private static Result Fail(Result result, string error, string message)
    {
        result.Success = false;
        result.Error = error;
        result.Message = message;
        return result;
    }

    private sealed class WindowsBackend : IDriverSetupBackend
    {
        private HMContext? _context;
        private HMContext Context => _context ??= new HMContext();

        public string? RuntimeVersion => typeof(HMContext).Assembly
            .GetCustomAttribute<AssemblyFileVersionAttribute>()?.Version;
        public bool IsElevated
        {
            get
            {
                using var identity = WindowsIdentity.GetCurrent();
                return new WindowsPrincipal(identity).IsInRole(WindowsBuiltInRole.Administrator);
            }
        }
        public bool IsSunshineRunning
        {
            get
            {
                var processes = Process.GetProcessesByName("sunshine");
                try { return processes.Length != 0; }
                finally { foreach (var process in processes) process.Dispose(); }
            }
        }
        public bool IsDriverInstalled => Context.IsDriverInstalled;
        public bool IsUsbipAvailable => HMContext.IsUsbipBackendAvailable;
        public void InstallHidDriver() => Context.InstallDriver();
        public void InstallUsbipBackend() => HMContext.InstallUsbipBackend(Console.Error.WriteLine);
        public void Dispose() => _context?.Dispose();
    }
}
