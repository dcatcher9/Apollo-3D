using System.Buffers.Binary;
using System.IO.Pipes;
using System.Text.Json;
using HIDMaestro;

namespace Sunshine.Ds5Sidecar;

internal static class ProtocolSelfTest
{
    internal static async Task RunHostProtocolChecksAsync()
    {
        var pipeName = $"sunshine-device-host-contract-{Environment.ProcessId}-{Guid.NewGuid():N}";
        using var stopping = new CancellationTokenSource(TimeSpan.FromSeconds(10));
        await using var server = new VirtualDeviceHostServer(
            pipeName, skipOwnerVerificationForTests: true);
        var serverTask = server.RunAsync(stopping.Token);

        await using var client = new NamedPipeClientStream(
            ".", pipeName, PipeDirection.InOut,
            PipeOptions.Asynchronous | PipeOptions.WriteThrough);
        await client.ConnectAsync(5_000, stopping.Token);

        await SendAsync(client, new Protocol.Message(
            Protocol.MessageType.Hello, 1, new byte[4]), stopping.Token);
        var hello = await ReceiveUntilAsync(
            client, Protocol.MessageType.HelloReply, 1, stopping.Token);
        Require(hello.Payload.Length == 4, "host protocol hello reply");

        // Message 12 belonged to the removed microphone prototype. It is refused
        // like every other message the helper does not implement.
        await SendAsync(client, new Protocol.Message(
            (Protocol.MessageType)12, 2, new byte[8]), stopping.Token);
        var refused = await ReceiveAsync(client, stopping.Token);
        Require(refused.Type == Protocol.MessageType.Error && refused.RequestId == 2,
            "host protocol unsupported message rejection");

        client.Dispose();
        await serverTask.WaitAsync(TimeSpan.FromSeconds(5));
    }

    internal static async Task<int> RunAsync(bool composite, string? resultPath, string? audioWriterPath)
    {
        RunDeterministicChecks();
        VerifyAdaptiveTriggerEncoding();
        var pipeName = $"sunshine-ds5-self-test-{Environment.ProcessId}-{Guid.NewGuid():N}";
        using var stopping = new CancellationTokenSource(TimeSpan.FromSeconds(30));
        await using var server = new VirtualDeviceHostServer(pipeName);
        var serverTask = server.RunAsync(stopping.Token);

        await using var client = new NamedPipeClientStream(
            ".", pipeName, PipeDirection.InOut, PipeOptions.Asynchronous | PipeOptions.WriteThrough);
        await client.ConnectAsync(10_000, stopping.Token);

        await SendAsync(client, new Protocol.Message(Protocol.MessageType.Hello, 1, new byte[4]), stopping.Token);
        var hello = await ReceiveAsync(client, stopping.Token);
        Require(hello.Type == Protocol.MessageType.HelloReply && hello.RequestId == 1 && hello.Payload.Length == 4,
            "hello reply");
        var helloCapabilities = (Protocol.Capability)BinaryPrimitives.ReadUInt32LittleEndian(hello.Payload);
        Require(helloCapabilities.HasFlag(Protocol.Capability.Hid), "hello HID capability");
        Require(helloCapabilities.HasFlag(Protocol.Capability.AdaptiveTriggers), "hello adaptive trigger capability");
        Require(helloCapabilities.HasFlag(Protocol.Capability.AudioPolicyViolation),
            "hello audio endpoint policy capability");

        await SendAsync(client, new Protocol.Message(
            Protocol.MessageType.Attach, 2, new byte[] { 0, 0, composite ? (byte)1 : (byte)0, 0 }), stopping.Token);
        var attach = await ReceiveAsync(client, stopping.Token);
        if (attach.Type == Protocol.MessageType.Error)
            throw new InvalidOperationException(DecodeError(attach.Payload));
        Require(attach.Type == Protocol.MessageType.AttachReply && attach.RequestId == 2 && attach.Payload.Length == 8,
            "attach reply");
        var capabilities = (Protocol.Capability)BinaryPrimitives.ReadUInt32LittleEndian(attach.Payload.AsSpan(4));
        Require(capabilities.HasFlag(Protocol.Capability.Hid), "HID capability");
        Require(capabilities.HasFlag(Protocol.Capability.AdaptiveTriggers), "adaptive trigger capability");
        Require(!composite || capabilities.HasFlag(Protocol.Capability.AudioFourChannel),
            "composite four-channel audio capability");

        var input = new byte[20];
        input[0] = 0;
        BinaryPrimitives.WriteUInt32LittleEndian(input.AsSpan(4), 0x1000 | 0x0010); // Cross + Start
        input[8] = 64;
        input[9] = 128;
        BinaryPrimitives.WriteInt16LittleEndian(input.AsSpan(12), -1234);
        BinaryPrimitives.WriteInt16LittleEndian(input.AsSpan(14), 2345);
        await SendAsync(client, new Protocol.Message(Protocol.MessageType.InputState, 0, input), stopping.Token);

        var touch = new byte[20];
        touch[0] = 0;
        touch[1] = 1;
        BinaryPrimitives.WriteUInt32LittleEndian(touch.AsSpan(4), 42);
        WriteFloat(touch.AsSpan(8), 0.25f);
        WriteFloat(touch.AsSpan(12), 0.75f);
        WriteFloat(touch.AsSpan(16), 1.0f);
        await SendAsync(client, new Protocol.Message(Protocol.MessageType.Touch, 0, touch), stopping.Token);
        touch[1] = 2;
        await SendAsync(client, new Protocol.Message(Protocol.MessageType.Touch, 0, touch), stopping.Token);
        touch[1] = 5; // LI_TOUCH_EVENT_BUTTON_ONLY must not mutate contact state or fail.
        await SendAsync(client, new Protocol.Message(Protocol.MessageType.Touch, 0, touch), stopping.Token);

        var motion = new byte[16];
        motion[0] = 0;
        motion[1] = 1;
        WriteFloat(motion.AsSpan(4), 0.0f);
        WriteFloat(motion.AsSpan(8), 9.80665f);
        WriteFloat(motion.AsSpan(12), 0.0f);
        await SendAsync(client, new Protocol.Message(Protocol.MessageType.Motion, 0, motion), stopping.Token);

        await SendAsync(client, new Protocol.Message(
            Protocol.MessageType.Battery, 0, new byte[] { 0, 3, 80, 0 }), stopping.Token);

        var capturedHapticsBytes = 0;
        if (!string.IsNullOrWhiteSpace(audioWriterPath))
        {
            Require(composite, "audio writer requires composite profile");
            Require(File.Exists(audioWriterPath), "audio writer executable");
            await Task.Delay(500, stopping.Token);
            using var writer = System.Diagnostics.Process.Start(new System.Diagnostics.ProcessStartInfo
            {
                FileName = audioWriterPath,
                WorkingDirectory = Path.GetDirectoryName(audioWriterPath)!,
                UseShellExecute = false,
                CreateNoWindow = true,
            }) ?? throw new InvalidOperationException("Unable to launch the haptics audio writer");
            while (capturedHapticsBytes == 0)
            {
                var haptics = await ReceiveUntilTypeAsync(client, Protocol.MessageType.HapticsPcm, stopping.Token);
                Require(haptics.Payload.Length >= 24 && haptics.Payload[3] == 2 && haptics.Payload[6] == 16,
                    "haptics PCM format");
                var frameCount = BinaryPrimitives.ReadUInt16LittleEndian(haptics.Payload.AsSpan(4));
                Require(BinaryPrimitives.ReadUInt32LittleEndian(haptics.Payload.AsSpan(20)) == 48000 &&
                        haptics.Payload.Length == 24 + frameCount * 4,
                    "haptics PCM size");
                var energy = 0L;
                for (var i = 24; i + 1 < haptics.Payload.Length; i += 2)
                    energy += Math.Abs((int)BinaryPrimitives.ReadInt16LittleEndian(haptics.Payload.AsSpan(i, 2)));
                if (energy != 0)
                    capturedHapticsBytes = haptics.Payload.Length - 24;
            }
            await writer.WaitForExitAsync(stopping.Token);
            Require(writer.ExitCode == 0, "audio writer exit code");
        }

        await SendAsync(client, new Protocol.Message(
            Protocol.MessageType.Detach, 3, new byte[] { 0 }), stopping.Token);
        var detach = await ReceiveUntilAsync(client, Protocol.MessageType.DetachReply, 3, stopping.Token);
        Require(detach.Payload.Length == 1 && detach.Payload[0] == 0, "detach reply");

        // Reattach and intentionally drop the owner pipe without DETACH. Completion
        // of serverTask proves the EOF path disposed the still-attached controller.
        await SendAsync(client, new Protocol.Message(
            Protocol.MessageType.Attach, 4, new byte[] { 0, 0, composite ? (byte)1 : (byte)0, 0 }), stopping.Token);
        var ownerCleanupAttach = await ReceiveUntilAsync(
            client, Protocol.MessageType.AttachReply, 4, stopping.Token);
        Require(ownerCleanupAttach.Payload.Length == 8, "owner cleanup attach reply");

        client.Dispose();
        await serverTask.WaitAsync(TimeSpan.FromSeconds(10));

        var result = JsonSerializer.Serialize(new
        {
            protocol = Protocol.Version,
            profile = composite ? "dualsense-composite" : "dualsense",
            hello = true,
            attached = true,
            four_channel_audio = capabilities.HasFlag(Protocol.Capability.AudioFourChannel),
            input = true,
            touch = true,
            motion = true,
            battery = true,
            adaptive_triggers = true,
            haptics_pcm = capturedHapticsBytes != 0,
            haptics_bytes = capturedHapticsBytes,
            detached = true,
            owner_disconnect_cleanup = true,
            cleanup = true,
        });
        Console.WriteLine(result);
        if (!string.IsNullOrWhiteSpace(resultPath))
            await File.WriteAllTextAsync(resultPath, result, stopping.Token);
        return 0;
    }

    internal static void RunDeterministicChecks()
    {
        VerifyProtocolAbi();
        VerifyBundledCompositeProfile();
        VerifyProfileSelection();
        VerifyHapticsChannelIsolation();
        VerifyHapticsSampleTimeline();
        VerifyHapticsStreamLifecycle();
        VerifyHapticsStopOrdering();
        VerifyDefaultAudioEndpointClassification();
        VerifyDefaultAudioEndpointPolicy();
        VerifyControllerStateSubmissionPolicy();
        VerifySensorTimestampEncoding();
        VerifyOutputValidityFlags();
        VerifyOutputValidityGating();
        DriverSetupSelfTest.Run();
    }

    private static void VerifyProtocolAbi()
    {
        Require(Protocol.Magic == 0x35534453 && Protocol.Version == 1 && Protocol.HeaderSize == 16,
            "SDS5 v1 header constants");
        Require((ushort)Protocol.MessageType.Attach == 3 &&
                (ushort)Protocol.MessageType.Shutdown == 11 &&
                (ushort)Protocol.MessageType.Rumble == 101 &&
                (ushort)Protocol.MessageType.AudioPolicyViolation == 105 &&
                (ushort)Protocol.MessageType.Error == 255,
            "SDS5 v1 message numbers");

        var attachPayload = new byte[] { 0x02, 0x01, 0x01, 0x00 };
        var frame = Protocol.Encode(new Protocol.Message(
            Protocol.MessageType.Attach, 0x11223344, attachPayload));
        Require(frame.SequenceEqual(new byte[]
        {
            0x53, 0x44, 0x53, 0x35, 0x01, 0x00, 0x03, 0x00,
            0x04, 0x00, 0x00, 0x00, 0x44, 0x33, 0x22, 0x11,
            0x02, 0x01, 0x01, 0x00,
        }), "attach golden frame");
        var header = Protocol.DecodeHeader(frame.AsSpan(0, Protocol.HeaderSize));
        Require(header.Type == Protocol.MessageType.Attach && header.PayloadLength == 4 &&
                header.RequestId == 0x11223344,
            "attach golden header decode");

        var wrongMagic = frame[..Protocol.HeaderSize];
        wrongMagic[0] ^= 0xff;
        RequireInvalidData(() => Protocol.DecodeHeader(wrongMagic), "header magic rejection");
        var oversized = frame[..Protocol.HeaderSize];
        BinaryPrimitives.WriteUInt32LittleEndian(oversized.AsSpan(8, 4), Protocol.MaxPayloadSize + 1u);
        RequireInvalidData(() => Protocol.DecodeHeader(oversized), "header payload limit rejection");
    }

    private static void RequireInvalidData(Action action, string operation)
    {
        try
        {
            action();
            Require(false, operation);
        }
        catch (InvalidDataException)
        {
            // Expected.
        }
    }

    private static void VerifySensorTimestampEncoding()
    {
        Require(ControllerSession.EncodeSensorTimestamp(0) == 0,
            "DualSense sensor timestamp origin");
        Require(ControllerSession.EncodeSensorTimestamp(1_000_000) == 3_000_000,
            "DualSense sensor timestamp tick rate");
        Require(ControllerSession.EncodeSensorTimestamp(0x55555556) == 2,
            "DualSense sensor timestamp rollover");
    }

    private static void VerifyProfileSelection()
    {
        Require(DeviceRegistry.SelectProfileId(1, 0, true) == DualSenseHapticsAudio.CompositeProfileId,
            "composite attach profile selection");
        Require(DeviceRegistry.SelectProfileId(0, 0, true) == "dualsense",
            "HID attach profile selection");
        // Bit 0 once selected a removed compatibility identity; no attach flag is defined now.
        RequireInvalidData(() => DeviceRegistry.SelectProfileId(1, 1, true), "attach flag rejection");
    }

    private static void VerifyControllerStateSubmissionPolicy()
    {
        var policy = new ControllerStateSubmissionPolicy();
        Require(!policy.ObserveInput(0, true), "idle controller state coalescing");
        Require(policy.ObserveInput(0, false), "analog activation boundary");
        Require(!policy.ObserveInput(0, false), "continuous analog state coalescing");
        Require(policy.ObserveInput(0, true), "analog neutral boundary");
        Require(policy.ObserveInput(0x1000, true), "button press boundary");
        Require(policy.ObserveInput(0, true), "button release boundary");
    }

    private static void VerifyDefaultAudioEndpointClassification()
    {
        Require(Enum.GetUnderlyingType(typeof(DefaultAudioEndpointGuard.AudioRole)) == typeof(int),
            "default audio role COM width");
        var virtualDualSense = new[]
        {
            new DefaultAudioEndpointGuard.DeviceNodeIdentity(
                @"SWD\MMDEVAPI\{0.0.0.00000000}.fixture", Array.Empty<string>()),
            new DefaultAudioEndpointGuard.DeviceNodeIdentity(
                @"USB\VID_054C&PID_0CE6&MI_00\fixture",
                new[] { @"USB\VID_054C&PID_0CE6&MI_00" }),
            new DefaultAudioEndpointGuard.DeviceNodeIdentity(
                @"ROOT\USB\0000", new[] { @"ROOT\HIDMAESTRO_UDE" }),
        };
        Require(DefaultAudioEndpointGuard.IsVirtualDualSenseChain(virtualDualSense),
            "virtual HIDMaestro DualSense endpoint classification");

        Require(!DefaultAudioEndpointGuard.IsVirtualDualSenseChain(virtualDualSense[..2]),
            "physical DualSense endpoint exclusion");
        Require(!DefaultAudioEndpointGuard.IsVirtualDualSenseChain(new[]
        {
            new DefaultAudioEndpointGuard.DeviceNodeIdentity(
                @"ROOT\USB\0000", new[] { @"ROOT\HIDMAESTRO_UDE" }),
        }), "unrelated HIDMaestro endpoint exclusion");
    }

    private static void VerifyDefaultAudioEndpointPolicy()
    {
        Require(DefaultAudioEndpointPolicy.NeedsUpdate(null, null),
            "missing default audio endpoint policy");
        Require(DefaultAudioEndpointPolicy.NeedsUpdate(
                "{00000000-0000-0000-0000-000000000000}", 0x00000101),
            "partial default audio endpoint policy");
        Require(!DefaultAudioEndpointPolicy.NeedsUpdate(
                "{00000000-0000-0000-0000-000000000000}", 0x00000307),
            "complete default audio endpoint policy");
        var quadraphonic = DualSenseSpeakerConfiguration.CreateQuadraphonicFormat();
        Require(DualSenseSpeakerConfiguration.HasValidInteropLayout(),
            "Core Audio interop ABI");
        Require(DualSenseSpeakerConfiguration.IsQuadraphonic(quadraphonic),
            "quadraphonic Core Audio speaker configuration");
        quadraphonic.ChannelMask = 0x00000003;
        Require(!DualSenseSpeakerConfiguration.IsQuadraphonic(quadraphonic),
            "stereo Core Audio speaker configuration rejection");
    }

    private static void VerifyBundledCompositeProfile()
    {
        using var stream = typeof(ProtocolSelfTest).Assembly.GetManifestResourceStream(
            "Sunshine.Ds5Sidecar.profiles.dualsense-composite.json")
            ?? throw new InvalidOperationException("Bundled composite profile is missing");
        using var memory = new MemoryStream();
        stream.CopyTo(memory);
        var profileJson = DualSenseHapticsAudio.CreateRuntimeCompositeProfile(memory.ToArray());
        DualSenseHapticsAudio.ValidateCompositeProfile(profileJson);

        var profileText = System.Text.Encoding.UTF8.GetString(profileJson);
        RequireProfileRejected(profileText.Replace("\"channels\":4", "\"channels\":2", StringComparison.Ordinal),
            "stereo composite profile rejection");
        RequireProfileRejected(profileText.Replace(
                "\"hapticLeft\",\"hapticRight\"",
                "\"hapticRight\",\"hapticLeft\"",
                StringComparison.Ordinal),
            "swapped haptics role rejection");
        RequireProfileRejected(profileText.Replace(
                "\"volumeCurRaw\":0",
                "\"volumeCurRaw\":-25600",
                StringComparison.Ordinal),
            "muted speaker control rejection");
    }

    private static void RequireProfileRejected(string profileJson, string operation)
    {
        try
        {
            DualSenseHapticsAudio.ValidateCompositeProfile(System.Text.Encoding.UTF8.GetBytes(profileJson));
            Require(false, operation);
        }
        catch (InvalidDataException)
        {
            // Expected.
        }
    }

    private static void VerifyHapticsChannelIsolation()
    {
        var frames = new byte[DualSenseHapticsAudio.InputFrameBytes * 2];
        WriteSample(frames, 0, 1234);
        WriteSample(frames, 1, -2345);
        WriteSample(frames, 4, short.MaxValue);
        WriteSample(frames, 5, short.MinValue);
        var speakerOnly = DualSenseHapticsAudio.Extract(frames);
        Require(speakerOnly.AsSpan().IndexOfAnyExcept((byte)0) == -1,
            "speaker channels cannot leak into haptics");

        WriteSample(frames, 2, 3456);
        WriteSample(frames, 3, -4567);
        WriteSample(frames, 6, 5678);
        WriteSample(frames, 7, -6789);
        var haptics = DualSenseHapticsAudio.Extract(frames);
        Require(BinaryPrimitives.ReadInt16LittleEndian(haptics.AsSpan(0, 2)) == 3456 &&
                BinaryPrimitives.ReadInt16LittleEndian(haptics.AsSpan(2, 2)) == -4567 &&
                BinaryPrimitives.ReadInt16LittleEndian(haptics.AsSpan(4, 2)) == 5678 &&
                BinaryPrimitives.ReadInt16LittleEndian(haptics.AsSpan(6, 2)) == -6789,
            "haptics channels preserve exact samples");

        try
        {
            DualSenseHapticsAudio.Extract(frames.AsSpan(0, frames.Length - 1));
            Require(false, "incomplete four-channel frame rejection");
        }
        catch (InvalidDataException)
        {
            // Expected: arbitrary callback fragmentation is reassembled by
            // DualSenseHapticsStream before complete frames reach the extractor.
        }
    }

    private static ulong HapticsTime(Protocol.Message message) =>
        BinaryPrimitives.ReadUInt64LittleEndian(message.Payload.AsSpan(12, 8));

    private static uint HapticsSequence(Protocol.Message message) =>
        BinaryPrimitives.ReadUInt32LittleEndian(message.Payload.AsSpan(8, 4));

    private static void VerifyHapticsSampleTimeline()
    {
        var messages = new List<Protocol.Message>();
        long clock = 1_000_000;
        using var stream = new DualSenseHapticsStream(3, 2, message =>
        {
            messages.Add(message);
            clock += 777; // Emission cost must not become inter-chunk timing.
        }, () => clock);
        stream.OnAudioStreamingChanged(null, true);
        var audio = new byte[600 * DualSenseHapticsAudio.InputFrameBytes];
        for (var frame = 0; frame < 600; frame++)
        {
            WriteSample(audio, frame * 4, short.MaxValue);
            WriteSample(audio, frame * 4 + 1, short.MinValue);
            WriteSample(audio, frame * 4 + 2, (short)frame);
            WriteSample(audio, frame * 4 + 3, (short)-frame);
        }
        // Invoke the same callback subscribed directly to HIDMaestro's batched
        // OUT URBs. This covers extraction, chunking, clocking, and the wire.
        stream.OnAudioFrames(null, audio);
        Require(messages.Count == 3, "batched haptics split at 240 frames");
        var offset = 0;
        for (var chunk = 0; chunk < 3; chunk++)
        {
            var message = messages[chunk];
            var payload = message.Payload;
            var frames = chunk < 2 ? 240 : 120;
            Require(message.Type == Protocol.MessageType.HapticsPcm &&
                    payload[0] == 3 && payload[1] == 2 && payload[3] == 2 && payload[6] == 16 &&
                    BinaryPrimitives.ReadUInt16LittleEndian(payload.AsSpan(4, 2)) == frames &&
                    BinaryPrimitives.ReadUInt32LittleEndian(payload.AsSpan(20, 4)) == 48000 &&
                    payload.Length == 24 + frames * 4,
                "batched haptics exact wire shape");
            Require(HapticsTime(message) == 1_000_000ul + (ulong)chunk * 5_000 &&
                    HapticsSequence(message) == chunk &&
                    payload[2] == (chunk == 0 ? (byte)Protocol.HapticsFlags.StreamStart : 0),
                "consecutive chunks have non-overlapping 5ms presentation times and one START");
            for (var frame = 0; frame < frames; frame++)
            {
                Require(BinaryPrimitives.ReadInt16LittleEndian(payload.AsSpan(24 + frame * 4, 2)) == offset + frame &&
                        BinaryPrimitives.ReadInt16LittleEndian(payload.AsSpan(26 + frame * 4, 2)) == -(offset + frame),
                    "batched haptics retain only the exact actuator samples");
            }
            offset += frames;
        }
        clock += 10_000_000; // Subsequent callback jitter cannot move the sample cursor.
        stream.OnAudioStreamingChanged(null, true); // Duplicate start is not a new stream.
        for (var frame = 0; frame < 48; frame++)
        {
            stream.OnAudioFrames(null, new byte[DualSenseHapticsAudio.InputFrameBytes]);
            Require(HapticsTime(messages[^1]) == 1_000_000ul + (600ul + (uint)frame) * 1_000_000 / 48_000,
                "fractional microseconds use cumulative frames without rounding drift");
        }
        stream.OnAudioStreamingChanged(null, false);
        Require(HapticsTime(messages[^1]) == 1_013_500 &&
                messages[^1].Payload[2] == (byte)Protocol.HapticsFlags.StreamEnd,
            "END lands exactly after the last emitted sample");
    }

    private static void VerifyHapticsStreamLifecycle()
    {
        var messages = new List<Protocol.Message>();
        long clock = 100;
        using var stream = new DualSenseHapticsStream(0, 0, messages.Add, () => clock);
        stream.OnAudioStreamingChanged(null, false);
        stream.InitializeStreamingState(true); // Stale startup snapshot loses to a notification.
        stream.OnAudioFrames(null, new byte[8]);
        Require(messages.Count == 0, "stale audio startup snapshot cannot resurrect a stopped stream");
        stream.OnAudioStreamingChanged(null, true);
        stream.OnAudioFrames(null, new byte[] { 0x99, 0x88, 0x77 });
        Require(messages.Count == 0, "partial source frame waits for its remaining bytes");
        stream.OnAudioStreamingChanged(null, false);
        Require(messages.Count == 1 && messages[0].Payload[2] == (byte)Protocol.HapticsFlags.StreamEnd,
            "stopping a partial-only stream emits an empty END");
        stream.OnAudioStreamingChanged(null, false);
        stream.OnAudioFrames(null, new byte[16]);
        Require(messages.Count == 1, "stopped stream ignores late frames and duplicate END");

        clock = 2_000_000;
        stream.OnAudioStreamingChanged(null, true);
        var source = new byte[] { 9, 8, 7, 6, 0x34, 0x12, 0x78, 0x56 };
        stream.OnAudioFrames(null, source.AsMemory(0, 3));
        stream.OnAudioFrames(null, source.AsMemory(3));
        Require(messages.Count == 2 && HapticsTime(messages[1]) == 2_000_000 &&
                HapticsSequence(messages[1]) == 1 &&
                messages[1].Payload[2] == (byte)Protocol.HapticsFlags.StreamStart &&
                messages[1].Payload.AsSpan(24).SequenceEqual(source.AsSpan(4)),
            "restart discards old residual, reanchors time, and retains controller sequence");
        stream.Dispose();
        Require(messages.Count == 3 && HapticsTime(messages[2]) == 2_000_020 &&
                HapticsSequence(messages[2]) == 2 &&
                messages[2].Payload[2] == (byte)Protocol.HapticsFlags.StreamEnd,
            "disposing an active stream emits the final sample boundary");
        stream.OnAudioStreamingChanged(null, true);
        stream.OnAudioFrames(null, source);
        stream.Dispose();
        Require(messages.Count == 3, "disposed haptics stream cannot deliver late callbacks");
    }

    private static void VerifyHapticsStopOrdering()
    {
        var messages = new List<Protocol.Message>();
        using var firstChunk = new ManualResetEventSlim();
        using var releaseChunk = new ManualResetEventSlim();
        using var stopStarted = new ManualResetEventSlim();
        using var stream = new DualSenseHapticsStream(0, 0, message =>
        {
            messages.Add(message);
            if (messages.Count == 1)
            {
                firstChunk.Set();
                Require(releaseChunk.Wait(TimeSpan.FromSeconds(5)), "release blocked audio chunk");
            }
        }, () => 1000);
        stream.OnAudioStreamingChanged(null, true);
        var framesTask = Task.Factory.StartNew(
            () => stream.OnAudioFrames(null, new byte[480 * 8]),
            CancellationToken.None, TaskCreationOptions.LongRunning, TaskScheduler.Default);
        Require(firstChunk.Wait(TimeSpan.FromSeconds(5)), "audio batch reaches first chunk");
        var stopTask = Task.Factory.StartNew(() =>
        {
            stopStarted.Set();
            stream.OnAudioStreamingChanged(null, false);
        }, CancellationToken.None, TaskCreationOptions.LongRunning, TaskScheduler.Default);
        try
        {
            Require(stopStarted.Wait(TimeSpan.FromSeconds(5)) && !stopTask.Wait(50),
                "concurrent END waits for the whole in-flight USB window");
        }
        finally
        {
            releaseChunk.Set();
            Require(Task.WaitAll(new[] { framesTask, stopTask }, TimeSpan.FromSeconds(5)),
                "audio batch and concurrent END finish without deadlock");
        }
        Require(messages.Count == 3 && messages[1].Payload[2] == 0 &&
                messages[2].Payload[2] == (byte)Protocol.HapticsFlags.StreamEnd &&
                HapticsTime(messages[1]) == 6000 && HapticsTime(messages[2]) == 11000,
            "END follows the final chunk in sequence and sample time");
    }

    private static void WriteSample(Span<byte> frames, int sample, short value) =>
        BinaryPrimitives.WriteInt16LittleEndian(frames.Slice(sample * 2, 2), value);

    private static async Task SendAsync(Stream stream, Protocol.Message message, CancellationToken cancellationToken)
    {
        var frame = Protocol.Encode(message);
        await stream.WriteAsync(frame, cancellationToken);
        await stream.FlushAsync(cancellationToken);
    }

    private static async Task<Protocol.Message> ReceiveAsync(Stream stream, CancellationToken cancellationToken)
    {
        var headerBytes = new byte[Protocol.HeaderSize];
        await stream.ReadExactlyAsync(headerBytes, cancellationToken);
        var header = Protocol.DecodeHeader(headerBytes);
        var payload = new byte[header.PayloadLength];
        if (payload.Length != 0)
            await stream.ReadExactlyAsync(payload, cancellationToken);
        return new Protocol.Message(header.Type, header.RequestId, payload);
    }

    private static async Task<Protocol.Message> ReceiveUntilAsync(
        Stream stream, Protocol.MessageType type, uint requestId, CancellationToken cancellationToken,
        Action<Protocol.Message>? observed = null)
    {
        while (true)
        {
            var message = await ReceiveAsync(stream, cancellationToken);
            observed?.Invoke(message);
            if (message.Type == Protocol.MessageType.Error && message.RequestId == requestId)
                throw new InvalidOperationException(DecodeError(message.Payload));
            if (message.Type == type && message.RequestId == requestId)
                return message;
        }
    }

    private static async Task<Protocol.Message> ReceiveUntilTypeAsync(
        Stream stream, Protocol.MessageType type, CancellationToken cancellationToken)
    {
        while (true)
        {
            var message = await ReceiveAsync(stream, cancellationToken);
            if (message.Type == Protocol.MessageType.Error)
                throw new InvalidOperationException(DecodeError(message.Payload));
            if (message.Type == type)
                return message;
        }
    }

    private static string DecodeError(ReadOnlySpan<byte> payload)
    {
        if (payload.Length < 8) return "Malformed sidecar error";
        var length = Math.Min(BinaryPrimitives.ReadUInt32LittleEndian(payload.Slice(4, 4)), (uint)(payload.Length - 8));
        return System.Text.Encoding.UTF8.GetString(payload.Slice(8, (int)length));
    }

    private static void WriteFloat(Span<byte> destination, float value) =>
        BinaryPrimitives.WriteInt32LittleEndian(destination, BitConverter.SingleToInt32Bits(value));

    private static void VerifyAdaptiveTriggerEncoding()
    {
        var state = new AdaptiveTriggerState();
        var left = Enumerable.Range(0x20, AdaptiveTriggerState.EffectSize).Select(value => (byte)value).ToArray();
        var right = Enumerable.Range(0x40, AdaptiveTriggerState.EffectSize).Select(value => (byte)value).ToArray();

        Require(state.TryUpdate(new Dictionary<string, object> { ["leftTriggerEffect"] = left },
                                default, 3, 2, out var leftMessage),
            "left adaptive trigger update");
        Require(leftMessage.Type == Protocol.MessageType.AdaptiveTriggers &&
                leftMessage.Payload.Length == 26 &&
                leftMessage.Payload[0] == 3 && leftMessage.Payload[1] == 2 &&
                leftMessage.Payload[2] == AdaptiveTriggerState.LeftFlag &&
                leftMessage.Payload[3] == left[0] && leftMessage.Payload[4] == 0 &&
                leftMessage.Payload.AsSpan(6, 10).SequenceEqual(left.AsSpan(1, 10)),
            "left adaptive trigger encoding");

        Require(!state.TryUpdate(new Dictionary<string, object> { ["leftTriggerEffect"] = left },
                                 default, 3, 2, out _),
            "adaptive trigger duplicate suppression");

        Require(state.TryUpdate(new Dictionary<string, object> { ["rightTriggerEffect"] = right },
                                default, 3, 2, out var rightMessage) &&
                rightMessage.Payload[2] == AdaptiveTriggerState.RightFlag &&
                rightMessage.Payload[3] == left[0] && rightMessage.Payload[4] == right[0] &&
                rightMessage.Payload.AsSpan(16, 10).SequenceEqual(right.AsSpan(1, 10)),
            "right adaptive trigger encoding");

        Require(state.TryReset(3, 2, out var resetMessage) &&
                resetMessage.Payload[2] == (AdaptiveTriggerState.LeftFlag | AdaptiveTriggerState.RightFlag) &&
                resetMessage.Payload.AsSpan(3).IndexOfAnyExcept((byte)0) == -1,
            "adaptive trigger reset");
        Require(!state.TryReset(3, 2, out _), "adaptive trigger reset duplicate suppression");
    }

    private static void VerifyOutputValidityFlags()
    {
        var absent = OutputValidFlags.From(new Dictionary<string, object>());
        Require(absent.LeftTrigger && absent.RightTrigger && absent.Lightbar,
            "output validity fallback");

        var silent = Flags(0x00, 0x00);
        Require(!silent.LeftTrigger && !silent.RightTrigger && !silent.Lightbar,
            "output validity gates every field it governs");

        Require(Flags(0x0C, 0x00) is { LeftTrigger: true, RightTrigger: true },
            "output validity both triggers");
        Require(Flags(0x08, 0x00) is { LeftTrigger: true, RightTrigger: false },
            "output validity left trigger only");
        Require(Flags(0x04, 0x00) is { LeftTrigger: false, RightTrigger: true },
            "output validity right trigger only");
        Require(Flags(0x00, 0x44).Lightbar && !Flags(0x00, 0x40).Lightbar,
            "output validity lightbar control");

        // A byte the decoder hides governs nothing: exposing only the lightbar
        // byte must not start gating the triggers.
        var lightbarOnly = OutputValidFlags.From(new Dictionary<string, object>
        {
            ["validFlag1"] = (byte)0x40,
        });
        Require(!lightbarOnly.Lightbar &&
                lightbarOnly.LeftTrigger && lightbarOnly.RightTrigger,
            "output validity gates per byte");
    }

    private static void VerifyOutputValidityGating()
    {
        var effect = Enumerable.Range(0x22, AdaptiveTriggerState.EffectSize).Select(value => (byte)value).ToArray();
        var idle = new byte[AdaptiveTriggerState.EffectSize];
        var state = new AdaptiveTriggerState();

        // A report that programs one trigger leaves the other's bytes zero. The
        // trigger it is not programming keeps the effect the client holds. The
        // validFlag0 bits are spelled out so a wrong constant cannot make the
        // gate agree with itself.
        var rightOnly = new Dictionary<string, object>
        {
            ["leftTriggerEffect"] = idle,
            ["rightTriggerEffect"] = effect,
            ["validFlag0"] = (byte)0x04,
        };
        var leftOnly = new Dictionary<string, object>
        {
            ["leftTriggerEffect"] = effect,
            ["rightTriggerEffect"] = idle,
            ["validFlag0"] = (byte)0x08,
        };

        Require(state.TryUpdate(rightOnly, OutputValidFlags.From(rightOnly), 1, 0, out var right) &&
                right.Payload[2] == AdaptiveTriggerState.RightFlag &&
                right.Payload[4] == effect[0],
            "right trigger armed by its own report");
        Require(state.TryUpdate(leftOnly, OutputValidFlags.From(leftOnly), 1, 0, out var left) &&
                left.Payload[2] == AdaptiveTriggerState.LeftFlag &&
                left.Payload[3] == effect[0] && left.Payload[4] == effect[0],
            "left trigger armed without disturbing the right");
        Require(!state.TryUpdate(leftOnly, OutputValidFlags.From(leftOnly), 1, 0, out _),
            "an unprogrammed trigger is never released");

        // A report that says it is programming both triggers with no effect is
        // the game letting go, and it reaches the client as written.
        var releaseBoth = new Dictionary<string, object>
        {
            ["leftTriggerEffect"] = idle,
            ["rightTriggerEffect"] = idle,
            ["validFlag0"] = (byte)0x0C,
        };
        Require(state.TryUpdate(releaseBoth, OutputValidFlags.From(releaseBoth), 1, 0, out var release) &&
                release.Payload[2] == (AdaptiveTriggerState.LeftFlag | AdaptiveTriggerState.RightFlag) &&
                release.Payload.AsSpan(3).IndexOfAnyExcept((byte)0) == -1,
            "a programmed release reaches the client");
    }

    private static OutputValidFlags Flags(byte flag0, byte flag1) =>
        OutputValidFlags.From(new Dictionary<string, object>
        {
            ["validFlag0"] = flag0,
            ["validFlag1"] = flag1,
        });

    private static void Require(bool condition, string operation)
    {
        if (!condition) throw new InvalidOperationException($"Protocol self-test failed at {operation}");
    }
}
