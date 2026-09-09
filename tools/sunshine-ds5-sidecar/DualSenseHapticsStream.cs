using System.Buffers.Binary;

namespace Sunshine.Ds5Sidecar;

/// <summary>
/// Converts complete USB audio windows to sequenced actuator PCM. The USB
/// transport batches multiple milliseconds into one callback, so presentation
/// time follows the sample count rather than the time each chunk is emitted.
/// </summary>
internal sealed class DualSenseHapticsStream : IDisposable
{
    private const int FramesPerPacket = 240;
    private readonly object _lock = new();
    private readonly byte _deviceId;
    private readonly byte _controllerNumber;
    private readonly Action<Protocol.Message> _emit;
    private readonly Func<long> _elapsedMicroseconds;
    private byte[] _residual = Array.Empty<byte>();
    private uint _sequence;
    private ulong _baseTimeUs;
    private ulong _framesEmitted;
    private bool _hasTimeBase;
    private bool _needsStart;
    private bool _streaming;
    private bool _streamStateObserved;
    private bool _disposed;

    internal DualSenseHapticsStream(byte deviceId, byte controllerNumber,
                                   Action<Protocol.Message> emit, Func<long> elapsedMicroseconds)
    {
        _deviceId = deviceId;
        _controllerNumber = controllerNumber;
        _emit = emit;
        _elapsedMicroseconds = elapsedMicroseconds;
    }

    internal void OnAudioStreamingChanged(object? sender, bool streaming)
    {
        // HIDMaestro raises alternate-setting changes on its connection reader
        // and PCM on its audio pump. Serialize them so END cannot overtake the
        // tail of a batch, and an old residual cannot enter a restarted stream.
        lock (_lock)
        {
            _streamStateObserved = true;
            SetStreaming(streaming);
        }
    }

    internal void InitializeStreamingState(bool streaming)
    {
        lock (_lock)
        {
            // A notification after event registration is newer than the
            // constructor's snapshot, even if it arrived before this lock.
            if (!_streamStateObserved)
                SetStreaming(streaming);
        }
    }

    private void SetStreaming(bool streaming)
    {
        if (_disposed || _streaming == streaming)
            return;
        if (!streaming)
            EndStream();
        else
        {
            _streaming = true;
            _needsStart = true;
            _hasTimeBase = false;
            _framesEmitted = 0;
            _residual = Array.Empty<byte>();
        }
    }

    internal void OnAudioFrames(object? sender, ReadOnlyMemory<byte> pcm)
    {
        lock (_lock)
        {
            if (_disposed || !_streaming)
                return;

            var combined = new byte[_residual.Length + pcm.Length];
            _residual.CopyTo(combined, 0);
            pcm.Span.CopyTo(combined.AsSpan(_residual.Length));
            var sourceFrameBytes = DualSenseHapticsAudio.InputFrameBytes;
            var usableBytes = combined.Length - combined.Length % sourceFrameBytes;
            _residual = usableBytes == combined.Length ? Array.Empty<byte>() : combined[usableBytes..];

            if (usableBytes == 0)
                return;
            if (!_hasTimeBase)
            {
                _baseTimeUs = (ulong)Math.Max(0, _elapsedMicroseconds());
                _hasTimeBase = true;
            }

            var source = combined.AsSpan(0, usableBytes);
            var frameCount = usableBytes / sourceFrameBytes;
            for (var offset = 0; offset < frameCount;)
            {
                var frames = Math.Min(FramesPerPacket, frameCount - offset);
                var haptics = DualSenseHapticsAudio.Extract(
                    source.Slice(offset * sourceFrameBytes, frames * sourceFrameBytes));
                var flags = _needsStart ? Protocol.HapticsFlags.StreamStart : Protocol.HapticsFlags.None;
                _needsStart = false;
                Emit(haptics, (ushort)frames, flags);
                _framesEmitted += (uint)frames;
                offset += frames;
            }
        }
    }

    private ulong PresentationTimeUs => !_hasTimeBase
        ? (ulong)Math.Max(0, _elapsedMicroseconds())
        : _baseTimeUs + _framesEmitted / DualSenseHapticsAudio.SampleRateHz * 1_000_000 +
          _framesEmitted % DualSenseHapticsAudio.SampleRateHz * 1_000_000 / DualSenseHapticsAudio.SampleRateHz;

    private void Emit(ReadOnlySpan<byte> pcm, ushort frames, Protocol.HapticsFlags flags)
    {
        // id:u8, controller:u8, flags:u8, channels:u8, frames:u16,
        // bits:u8, reserved:u8, seq:u32, timestamp:u64, rate:u32, PCM
        var payload = new byte[24 + pcm.Length];
        payload[0] = _deviceId;
        payload[1] = _controllerNumber;
        payload[2] = (byte)flags;
        payload[3] = DualSenseHapticsAudio.OutputChannels;
        BinaryPrimitives.WriteUInt16LittleEndian(payload.AsSpan(4, 2), frames);
        payload[6] = DualSenseHapticsAudio.BitsPerSample;
        BinaryPrimitives.WriteUInt32LittleEndian(payload.AsSpan(8, 4), _sequence++);
        BinaryPrimitives.WriteUInt64LittleEndian(payload.AsSpan(12, 8), PresentationTimeUs);
        BinaryPrimitives.WriteUInt32LittleEndian(payload.AsSpan(20, 4), DualSenseHapticsAudio.SampleRateHz);
        pcm.CopyTo(payload.AsSpan(24));
        _emit(new Protocol.Message(Protocol.MessageType.HapticsPcm, 0, payload));
    }

    private void EndStream()
    {
        _streaming = false;
        _residual = Array.Empty<byte>();
        Emit(ReadOnlySpan<byte>.Empty, 0, Protocol.HapticsFlags.StreamEnd);
    }

    public void Dispose()
    {
        lock (_lock)
        {
            if (_disposed)
                return;
            if (_streaming)
                EndStream();
            _disposed = true;
        }
    }
}
