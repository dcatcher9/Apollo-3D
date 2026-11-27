using System.Buffers.Binary;
using System.Reflection;
using HIDMaestro;

namespace Sunshine.Ds5Sidecar;

internal sealed class DeviceRegistry : IDisposable
{
    internal const Protocol.Capability BaseCapabilities =
        Protocol.Capability.Hid |
        Protocol.Capability.Output |
        Protocol.Capability.Touchpad |
        Protocol.Capability.Motion |
        Protocol.Capability.Battery |
        Protocol.Capability.AdaptiveTriggers |
        Protocol.Capability.AudioPolicyViolation;

    private readonly HMContext _context = new();
    private readonly Dictionary<byte, ControllerSession> _controllers = new();
    private readonly Action<Protocol.Message> _emit;
    private readonly bool _authoredHapticsAvailable;
    private readonly bool _genshinCompatibilityAvailable;

    internal DeviceRegistry(Action<Protocol.Message> emit)
    {
        _emit = emit;
        var compositeProfileValidated = LoadPatchedProfiles();
        if (!compositeProfileValidated)
            throw new InvalidOperationException("Unable to load the validated DualSense profiles");
        _context.LoadDefaultProfiles();
        _authoredHapticsAvailable = compositeProfileValidated &&
                                    _context.GetProfile(DualSenseHapticsAudio.CompositeProfileId) is not null &&
                                    HMContext.IsUsbipBackendAvailable;
        _genshinCompatibilityAvailable = _authoredHapticsAvailable &&
                                         _context.GetProfile(
                                             DualSenseHapticsAudio.GenshinCompatibilityProfileId) is not null;
    }

    internal Protocol.Capability Capabilities =>
        BaseCapabilities |
        (_genshinCompatibilityAvailable
            ? Protocol.Capability.GenshinCompatibilityIdentity
            : 0) |
        (_authoredHapticsAvailable
            ? Protocol.Capability.AudioFourChannel | Protocol.Capability.AuthoredHapticsPcm
            : 0);

    internal static bool HasPinnedRuntime
    {
        get
        {
            var text = typeof(HMContext).Assembly
                .GetCustomAttribute<AssemblyFileVersionAttribute>()?.Version;
            return Version.TryParse(text, out var version) && version == new Version(1, 6, 2, 0);
        }
    }

    internal void Attach(uint requestId, ReadOnlySpan<byte> payload)
    {
        // id:u8, controller:u8, profile:u8 (0 HID, 1 composite), flags:u8
        if (payload.Length != 4)
            throw new InvalidDataException("Invalid attach payload");
        var deviceId = payload[0];
        if (_controllers.ContainsKey(deviceId))
            throw new InvalidOperationException("The requested DS5 device already exists");

        var profileId = SelectProfileId(
            payload[2], (Protocol.AttachFlags)payload[3],
            _authoredHapticsAvailable, _genshinCompatibilityAvailable);
        var profile = _context.GetProfile(profileId)
                      ?? throw new InvalidOperationException($"HIDMaestro profile '{profileId}' is missing");
        if (!profile.RequiresUsbipBackend && !_context.IsDriverInstalled)
            throw new InvalidOperationException("The HIDMaestro driver is not installed; install the optional component explicitly before enabling DualSense");
        if (profile.RequiresUsbipBackend && !HMContext.IsUsbipBackendAvailable)
            throw new InvalidOperationException("The USB/IP backend is not installed; install the optional component explicitly before enabling authored haptics");

        if (profile.RequiresUsbipBackend)
        {
            try
            {
                DefaultAudioEndpointPolicy.EnsureNeverDefault(
                    TimeSpan.Zero, includePhantom: true, out _);
            }
            catch (Exception error)
            {
                Console.Error.WriteLine($"Unable to preseed the DualSense audio endpoint policy: {error.Message}");
            }
        }

        var controller = _context.CreateController(profile);
        if (profile.RequiresUsbipBackend)
            controller = ApplyDefaultAudioEndpointPolicy(controller, profile);
        ControllerSession session;
        try
        {
            session = new ControllerSession(deviceId, payload[1], controller, profile, _emit);
        }
        catch
        {
            controller.Dispose();
            throw;
        }
        _controllers.Add(deviceId, session);

        var reply = new byte[8];
        reply[0] = deviceId;
        reply[1] = session.HasAudio ? (byte)1 : (byte)0;
        BinaryPrimitives.WriteUInt32LittleEndian(reply.AsSpan(4, 4),
            (uint)(Protocol.Capability.Hid |
                   Protocol.Capability.Output |
                   Protocol.Capability.Touchpad |
                   Protocol.Capability.Motion |
                   Protocol.Capability.Battery |
                   Protocol.Capability.AdaptiveTriggers |
                   (_genshinCompatibilityAvailable
                       ? Protocol.Capability.GenshinCompatibilityIdentity
                       : 0) |
                   (session.HasAudio
                       ? Protocol.Capability.AudioFourChannel | Protocol.Capability.AuthoredHapticsPcm
                       : 0)));
        _emit(new Protocol.Message(Protocol.MessageType.AttachReply, requestId, reply));
        session.StartDefaultAudioEndpointGuard();
    }

    internal void Detach(uint requestId, ReadOnlySpan<byte> payload)
    {
        if (payload.Length != 1)
            throw new InvalidDataException("Invalid detach payload");
        if (_controllers.Remove(payload[0], out var controller))
            controller.Dispose();
        _emit(new Protocol.Message(Protocol.MessageType.DetachReply, requestId, new[] { payload[0] }));
    }

    internal ControllerSession GetController(ReadOnlySpan<byte> payload)
    {
        if (payload.IsEmpty || !_controllers.TryGetValue(payload[0], out var controller))
            throw new InvalidOperationException("The requested DS5 device is not attached");
        return controller;
    }

    internal static string SelectProfileId(
        byte profileMode, Protocol.AttachFlags flags,
        bool authoredHapticsAvailable, bool genshinCompatibilityAvailable)
    {
        if ((flags & ~Protocol.AttachFlags.GenshinCompatibilityIdentity) != 0)
            throw new InvalidDataException("Unsupported DS5 attach flags");
        var genshinCompatibility = flags.HasFlag(
            Protocol.AttachFlags.GenshinCompatibilityIdentity);
        if (genshinCompatibility && profileMode != 1)
            throw new InvalidDataException("Genshin compatibility requires the composite profile");

        return profileMode switch
        {
            0 => "dualsense",
            1 when genshinCompatibility && genshinCompatibilityAvailable =>
                DualSenseHapticsAudio.GenshinCompatibilityProfileId,
            1 when genshinCompatibility =>
                throw new InvalidOperationException("Genshin compatibility identity is unavailable"),
            1 when authoredHapticsAvailable => DualSenseHapticsAudio.CompositeProfileId,
            1 => throw new InvalidOperationException("Validated DualSense four-channel audio is unavailable"),
            _ => throw new InvalidDataException("Unsupported DS5 profile mode"),
        };
    }

    private HMController ApplyDefaultAudioEndpointPolicy(HMController controller, HMProfile profile)
    {
        const int recreationLimit = 2;
        for (var recreation = 0; recreation <= recreationLimit; ++recreation)
        {
            bool changed;
            try
            {
                if (!DefaultAudioEndpointPolicy.EnsureNeverDefault(
                        TimeSpan.FromSeconds(3), includePhantom: false, out changed))
                {
                    Console.Error.WriteLine(
                        "Unable to find the virtual DualSense audio interface; runtime default-device guard remains active");
                    return controller;
                }
            }
            catch (Exception error)
            {
                Console.Error.WriteLine(
                    $"Unable to apply the DualSense audio endpoint policy: {error.Message}; " +
                    "runtime default-device guard remains active");
                return controller;
            }

            if (!changed)
            {
                ApplySpeakerConfiguration();
                return controller;
            }

            controller.Dispose();
            controller = _context.CreateController(profile);
        }

        Console.Error.WriteLine(
            "DualSense audio endpoint identity did not stabilize after policy provisioning; " +
            "runtime default-device guard remains active");
        ApplySpeakerConfiguration();
        return controller;
    }

    private static void ApplySpeakerConfiguration()
    {
        try
        {
            if (!DualSenseSpeakerConfiguration.Ensure(TimeSpan.FromSeconds(3), out var changed))
            {
                Console.Error.WriteLine(
                    "Unable to find the virtual DualSense render endpoint; " +
                    "complete its quadraphonic speaker configuration manually");
            }
            else
                Console.Error.WriteLine(changed
                    ? "Configured the virtual DualSense render endpoint as quadraphonic"
                    : "Reapplied the virtual DualSense quadraphonic speaker configuration");
        }
        catch (Exception error)
        {
            Console.Error.WriteLine(
                $"Unable to configure the virtual DualSense speakers: {error.Message}; " +
                "complete the quadraphonic speaker setup manually");
        }
    }

    private bool LoadPatchedProfiles()
    {
        // Register Sunshine's always-armed profile copies before the stock
        // catalog. HIDMaestro keeps the first profile for duplicate ids.
        try
        {
            var assembly = typeof(DeviceRegistry).Assembly;
            using var privateProfiles = new PrivateProfileDirectory();
            var directory = privateProfiles.Path;
            foreach (var id in new[] { "dualsense", "dualsense-composite" })
            {
                var resourceName = $"Sunshine.Ds5Sidecar.profiles.{id}.json";
                using var stream = assembly.GetManifestResourceStream(resourceName)
                    ?? throw new InvalidOperationException($"Embedded profile '{resourceName}' is missing");
                using var memory = new MemoryStream();
                stream.CopyTo(memory);
                var profileBytes = memory.ToArray();
                if (id == DualSenseHapticsAudio.CompositeProfileId)
                    profileBytes = DualSenseHapticsAudio.CreateRuntimeCompositeProfile(profileBytes);
                File.WriteAllBytes(Path.Combine(directory, id + ".json"), profileBytes);
                if (id == DualSenseHapticsAudio.CompositeProfileId)
                {
                    var compatibilityProfile =
                        DualSenseHapticsAudio.CreateGenshinCompatibilityProfile(profileBytes);
                    File.WriteAllBytes(
                        Path.Combine(directory, DualSenseHapticsAudio.GenshinCompatibilityProfileId + ".json"),
                        compatibilityProfile);
                }
            }
            if (_context.LoadProfilesFromDirectory(directory) < 3)
                throw new InvalidOperationException("Patched DualSense profiles did not fully register");
            return true;
        }
        catch (Exception error)
        {
            Console.Error.WriteLine($"Unable to load patched DualSense profiles: {error.Message}");
            return false;
        }
    }

    internal void ReleaseAllDevices()
    {
        Exception? cleanupError = null;
        foreach (var controller in _controllers.Values)
        {
            try
            {
                controller.Dispose();
            }
            catch (Exception error)
            {
                cleanupError ??= error;
            }
        }
        _controllers.Clear();
        if (cleanupError is not null)
            throw new InvalidOperationException("One or more virtual devices failed to dispose", cleanupError);
    }

    public void Dispose()
    {
        try { ReleaseAllDevices(); }
        finally { _context.Dispose(); }
    }
}
