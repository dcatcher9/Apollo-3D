using System.ComponentModel;
using System.Runtime.InteropServices;
using System.Security.AccessControl;
using System.Security.Principal;

namespace Sunshine.Ds5Sidecar;

// HIDMaestro 1.6.2 only accepts profiles through a directory. Create its
// temporary input atomically with a private ACL, then remove it after loading.
// Elevated profiles are inaccessible to the same user's medium-integrity token.
internal sealed class PrivateProfileDirectory : IDisposable
{
    internal string Path { get; }

    internal PrivateProfileDirectory()
    {
        using var identity = WindowsIdentity.GetCurrent();
        var elevated = new WindowsPrincipal(identity).IsInRole(WindowsBuiltInRole.Administrator);
        var security = new DirectorySecurity();
        security.SetSecurityDescriptorSddlForm(elevated
            ? "O:BAG:BAD:P(A;OICI;FA;;;SY)(A;OICI;FA;;;BA)S:(ML;OICI;NW;;;HI)"
            : $"O:{identity.User!.Value}D:P(A;OICI;FA;;;SY)(A;OICI;FA;;;{identity.User.Value})");
        var descriptor = security.GetSecurityDescriptorBinaryForm();
        var pinned = GCHandle.Alloc(descriptor, GCHandleType.Pinned);
        Path = System.IO.Path.Combine(System.IO.Path.GetTempPath(),
            $"sunshine-ds5-profiles-{Environment.ProcessId}-{Guid.NewGuid():N}");
        try
        {
            var attributes = new SecurityAttributes
            {
                Length = Marshal.SizeOf<SecurityAttributes>(),
                Descriptor = pinned.AddrOfPinnedObject(),
            };
            if (!CreateDirectory(Path, ref attributes))
                throw new Win32Exception(Marshal.GetLastWin32Error(), "Unable to create private DualSense profile directory");
        }
        finally { pinned.Free(); }
    }

    public void Dispose()
    {
        Directory.Delete(Path, recursive: true);
    }

    [StructLayout(LayoutKind.Sequential)]
    private struct SecurityAttributes
    {
        internal int Length;
        internal IntPtr Descriptor;
        [MarshalAs(UnmanagedType.Bool)] internal bool InheritHandle;
    }

    [DllImport("kernel32.dll", EntryPoint = "CreateDirectoryW", CharSet = CharSet.Unicode, SetLastError = true)]
    [return: MarshalAs(UnmanagedType.Bool)]
    private static extern bool CreateDirectory(string path, ref SecurityAttributes attributes);
}
