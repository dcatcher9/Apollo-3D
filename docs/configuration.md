# Configuration

@admonition{ Host authority | @htmlonly
By providing the host authority (URI + port), you can easily open each configuration option in the config UI.
<br>
<script src="configuration.js"></script>
<strong>Host authority: </strong> <input type="text" id="host-authority" value="localhost:47990">
@endhtmlonly
}

Apollo works with the default settings for most users. In some cases you may want to configure it further.

The default location for the configuration file is listed below. You can use another location if you
choose by passing the full configuration file path as the first argument when you start Apollo.

**Example**
```powershell
sunshine.exe C:\path\to\sunshine.conf
```

The default location of the `apps.json` is the same as the configuration file. You can use a custom
location by modifying the configuration file.

**Default Config Directory**

| Host    | Location                                        |
|---------|-------------------------------------------------|
| Windows | @code{}%ProgramFiles%\\Apollo\\config@endcode   |

Although the configuration UI is recommended, Apollo can also be configured manually by
editing the `conf` file in a text editor. Use the examples as reference.

## General

### locale

<table>
    <tr>
        <td>Description</td>
        <td colspan="2">
            The locale used for Apollo's user interface.
        </td>
    </tr>
    <tr>
        <td>Default</td>
        <td colspan="2">@code{}
            en
            @endcode</td>
    </tr>
    <tr>
        <td>Example</td>
        <td colspan="2">@code{}
            locale = en
            @endcode</td>
    </tr>
    <tr>
        <td rowspan="20">Choices</td>
        <td>bg</td>
        <td>Bulgarian</td>
    </tr>
    <tr>
        <td>cs</td>
        <td>Czech</td>
    </tr>
    <tr>
        <td>de</td>
        <td>German</td>
    </tr>
    <tr>
        <td>en</td>
        <td>English</td>
    </tr>
    <tr>
        <td>en_GB</td>
        <td>English (UK)</td>
    </tr>
    <tr>
        <td>en_US</td>
        <td>English (United States)</td>
    </tr>
    <tr>
        <td>es</td>
        <td>Spanish</td>
    </tr>
    <tr>
        <td>fr</td>
        <td>French</td>
    </tr>
    <tr>
        <td>it</td>
        <td>Italian</td>
    </tr>
    <tr>
        <td>ja</td>
        <td>Japanese</td>
    </tr>
    <tr>
        <td>ko</td>
        <td>Korean</td>
    </tr>
    <tr>
        <td>pl</td>
        <td>Polish</td>
    </tr>
    <tr>
        <td>pt</td>
        <td>Portuguese</td>
    </tr>
    <tr>
        <td>pt_BR</td>
        <td>Portuguese (Brazilian)</td>
    </tr>
    <tr>
        <td>ru</td>
        <td>Russian</td>
    </tr>
    <tr>
        <td>sv</td>
        <td>Swedish</td>
    </tr>
    <tr>
        <td>tr</td>
        <td>Turkish</td>
    </tr>
    <tr>
        <td>uk</td>
        <td>Ukranian</td>
    </tr>
    <tr>
        <td>zh</td>
        <td>Chinese (Simplified)</td>
    </tr>
    <tr>
        <td>zh_TW</td>
        <td>Chinese (Traditional)</td>
    </tr>
</table>

### sunshine_name

<table>
    <tr>
        <td>Description</td>
        <td colspan="2">
            The name displayed by Artemis.
        </td>
    </tr>
    <tr>
        <td>Default</td>
        <td colspan="2">PC hostname</td>
    </tr>
    <tr>
        <td>Example</td>
        <td colspan="2">@code{}
            sunshine_name = Apollo
            @endcode</td>
    </tr>
</table>

### min_log_level

<table>
    <tr>
        <td>Description</td>
        <td colspan="2">
            The minimum log level printed to standard out.
        </td>
    </tr>
    <tr>
        <td>Default</td>
        <td colspan="2">@code{}
            info
            @endcode</td>
    </tr>
    <tr>
        <td>Example</td>
        <td colspan="2">@code{}
            min_log_level = info
            @endcode</td>
    </tr>
    <tr>
        <td rowspan="7">Choices</td>
        <td>verbose</td>
        <td>All logging message.
            @attention{This may negatively affect streaming performance.}</td>
    </tr>
    <tr>
        <td>debug</td>
        <td>Debug log messages and higher.
            @attention{This may negatively affect streaming performance.}</td>
    </tr>
    <tr>
        <td>info</td>
        <td>Informational log messages and higher.</td>
    </tr>
    <tr>
        <td>warning</td>
        <td>Warning log messages and higher.</td>
    </tr>
    <tr>
        <td>error</td>
        <td>Error log messages and higher.</td>
    </tr>
    <tr>
        <td>fatal</td>
        <td>Only fatal log messages.</td>
    </tr>
    <tr>
        <td>none</td>
        <td>No log messages.</td>
    </tr>
</table>

### diagnostics

<table>
    <tr>
        <td>Description</td>
        <td colspan="2">
            Controls runtime performance diagnostics. These diagnostics retain concise frame-pacing,
            AI inference, SBS warp, encoder/network, and local-display measurements. When disabled,
            Apollo does not allocate GPU timing queries or perform diagnostic per-frame clock,
            counter, or filesystem work.
        </td>
    </tr>
    <tr>
        <td>Default</td>
        <td colspan="2"><code>disabled</code>. Development configurations may explicitly enable it.</td>
    </tr>
    <tr>
        <td>Example</td>
        <td colspan="2">@code{}
            diagnostics = enabled
            @endcode</td>
    </tr>
</table>

### global_prep_cmd

<table>
    <tr>
        <td>Description</td>
        <td colspan="2">
            A list of commands to be run before/after all applications.
            If any of the prep-commands fail, starting the application is aborted.
        </td>
    </tr>
    <tr>
        <td>Default</td>
        <td colspan="2">@code{}
            []
            @endcode</td>
    </tr>
    <tr>
        <td>Example</td>
        <td colspan="2">@code{}
            global_prep_cmd = [{"do":"nircmd.exe setdisplay 1280 720 32 144","elevated":true,"undo":"nircmd.exe setdisplay 2560 1440 32 144"}]
            @endcode</td>
    </tr>
</table>

### system_tray

<table>
    <tr>
        <td>Description</td>
        <td colspan="2">
            Show icon in system tray and display desktop notifications
        </td>
    </tr>
    <tr>
        <td>Default</td>
        <td colspan="2">@code{}
            enabled
            @endcode</td>
    </tr>
    <tr>
        <td>Example</td>
        <td colspan="2">@code{}
            system_tray = enabled
            @endcode</td>
    </tr>
</table>

### hide_tray_controls

<table>
    <tr><td>Description</td><td>Hide the Force Stop, Restart, and Quit actions from the system-tray menu.</td></tr>
    <tr><td>Default</td><td><code>disabled</code></td></tr>
</table>

### enable_pairing

<table>
    <tr><td>Description</td><td>Allow new Artemis devices to pair with Apollo.</td></tr>
    <tr><td>Default</td><td><code>enabled</code></td></tr>
</table>

### enable_discovery

<table>
    <tr><td>Description</td><td>Advertise Apollo for automatic discovery. When disabled, clients must add the host address manually.</td></tr>
    <tr><td>Default</td><td><code>enabled</code></td></tr>
</table>

## Input

### gamepad

<table>
    <tr>
        <td>Description</td>
        <td colspan="2">
            The type of gamepad to emulate on the host.
        </td>
    </tr>
    <tr>
        <td>Default</td>
        <td colspan="2">@code{}
            auto
            @endcode</td>
    </tr>
    <tr>
        <td>Example</td>
        <td colspan="2">@code{}
            gamepad = auto
            @endcode</td>
    </tr>
    <tr>
        <td rowspan="2">Choices</td>
        <td>ds4</td>
        <td>DualShock 4 controller (PS4)</td>
    </tr>
    <tr>
        <td>x360</td>
        <td>Xbox 360 controller</td>
    </tr>
</table>

### ds4_back_as_touchpad_click

<table>
    <tr>
        <td>Description</td>
        <td colspan="2">
            Allow Select/Back inputs to also trigger DS4 touchpad click. Useful for clients looking to
            emulate touchpad click on Xinput devices.
            @hint{Only applies when gamepad is set to ds4 manually. Unused in other gamepad modes.}
        </td>
    </tr>
    <tr>
        <td>Default</td>
        <td colspan="2">@code{}
            enabled
            @endcode</td>
    </tr>
    <tr>
        <td>Example</td>
        <td colspan="2">@code{}
            ds4_back_as_touchpad_click = enabled
            @endcode</td>
    </tr>
</table>

### motion_as_ds4

<table>
    <tr>
        <td>Description</td>
        <td colspan="2">
            If a client reports that a connected gamepad has motion sensor support, emulate it on the
            host as a DS4 controller.
            <br>
            <br>
            When disabled, motion sensors will not be taken into account during gamepad type selection.
            @hint{Only applies when gamepad is set to auto.}
        </td>
    </tr>
    <tr>
        <td>Default</td>
        <td colspan="2">@code{}
            enabled
            @endcode</td>
    </tr>
    <tr>
        <td>Example</td>
        <td colspan="2">@code{}
            motion_as_ds4 = enabled
            @endcode</td>
    </tr>
</table>

### touchpad_as_ds4

<table>
    <tr>
        <td>Description</td>
        <td colspan="2">
            If a client reports that a connected gamepad has a touchpad, emulate it on the host
            as a DS4 controller.
            <br>
            <br>
            When disabled, touchpad presence will not be taken into account during gamepad type selection.
            @hint{Only applies when gamepad is set to auto.}
        </td>
    </tr>
    <tr>
        <td>Default</td>
        <td colspan="2">@code{}
            enabled
            @endcode</td>
    </tr>
    <tr>
        <td>Example</td>
        <td colspan="2">@code{}
            touchpad_as_ds4 = enabled
            @endcode</td>
    </tr>
</table>

### back_button_timeout

<table>
    <tr>
        <td>Description</td>
        <td colspan="2">
            If the Back/Select button is held down for the specified number of milliseconds,
            a Home/Guide button press is emulated.
            @tip{If back_button_timeout < 0, then the Home/Guide button will not be emulated.}
        </td>
    </tr>
    <tr>
        <td>Default</td>
        <td colspan="2">@code{}
            -1
            @endcode</td>
    </tr>
    <tr>
        <td>Example</td>
        <td colspan="2">@code{}
            back_button_timeout = 2000
            @endcode</td>
    </tr>
</table>

### key_repeat_delay

<table>
    <tr>
        <td>Description</td>
        <td colspan="2">
            The initial delay, in milliseconds, before repeating keys. Controls how fast keys will
            repeat themselves.
        </td>
    </tr>
    <tr>
        <td>Default</td>
        <td colspan="2">@code{}
            500
            @endcode</td>
    </tr>
    <tr>
        <td>Example</td>
        <td colspan="2">@code{}
            key_repeat_delay = 500
            @endcode</td>
    </tr>
</table>

### key_repeat_frequency

<table>
    <tr>
        <td>Description</td>
        <td colspan="2">
            How often keys repeat every second.
            @tip{This configurable option supports decimals.}
        </td>
    </tr>
    <tr>
        <td>Default</td>
        <td colspan="2">@code{}
            24.9
            @endcode</td>
    </tr>
    <tr>
        <td>Example</td>
        <td colspan="2">@code{}
            key_repeat_frequency = 24.9
            @endcode</td>
    </tr>
</table>

### always_send_scancodes

<table>
    <tr>
        <td>Description</td>
        <td colspan="2">
            Sending scancodes enhances compatibility with games and apps but may result in incorrect keyboard input
            from certain clients that aren't using a US English keyboard layout.
            <br>
            <br>
            Enable if keyboard input is not working at all in certain applications.
            <br>
            <br>
            Disable if keys on the client are generating the wrong input on the host.
            @caution{Applies to Windows only.}
        </td>
    </tr>
    <tr>
        <td>Default</td>
        <td colspan="2">@code{}
            enabled
            @endcode</td>
    </tr>
    <tr>
        <td>Example</td>
        <td colspan="2">@code{}
            always_send_scancodes = enabled
            @endcode</td>
    </tr>
</table>

### high_resolution_scrolling

<table>
    <tr>
        <td>Description</td>
        <td colspan="2">
            When enabled, Apollo will pass through high resolution scroll events from Artemis clients.
            <br>
            This can be useful to disable for older applications that scroll too fast with high resolution scroll
            events.
        </td>
    </tr>
    <tr>
        <td>Default</td>
        <td colspan="2">@code{}
            enabled
            @endcode</td>
    </tr>
    <tr>
        <td>Example</td>
        <td colspan="2">@code{}
            high_resolution_scrolling = enabled
            @endcode</td>
    </tr>
</table>

### native_pen_touch

<table>
    <tr>
        <td>Description</td>
        <td colspan="2">
            When enabled, Apollo will pass through native pen/touch events from Artemis clients.
            <br>
            This can be useful to disable for older applications without native pen/touch support.
        </td>
    </tr>
    <tr>
        <td>Default</td>
        <td colspan="2">@code{}
            enabled
            @endcode</td>
    </tr>
    <tr>
        <td>Example</td>
        <td colspan="2">@code{}
            native_pen_touch = enabled
            @endcode</td>
    </tr>
</table>

### forward_rumble

<table>
    <tr><td>Description</td><td>Forward host controller-rumble messages to connected clients.</td></tr>
    <tr><td>Default</td><td><code>enabled</code></td></tr>
</table>

### keybindings

<table>
    <tr>
        <td>Description</td>
        <td colspan="2">
            Sometimes it may be useful to map keybindings. Wayland won't allow clients to capture the Win Key
            for example.
            @tip{See [virtual key codes](https://docs.microsoft.com/en-us/windows/win32/inputdev/virtual-key-codes)}
            @hint{keybindings needs to have a multiple of two elements.}
            @note{This option is not available in the UI. A PR would be welcome.}
        </td>
    </tr>
    <tr>
        <td>Default</td>
        <td colspan="2">@code{}
            [
              0x10, 0xA0,
              0x11, 0xA2,
              0x12, 0xA4
            ]
            @endcode</td>
    </tr>
    <tr>
        <td>Example</td>
        <td colspan="2">@code{}
            keybindings = [
              0x10, 0xA0,
              0x11, 0xA2,
              0x12, 0xA4,
              0x4A, 0x4B
            ]
            @endcode</td>
    </tr>
</table>

## Audio/Video

### audio_sink

<table>
    <tr>
        <td>Description</td>
        <td colspan="2">
            The name of the audio sink used for audio loopback.
            @tip{To find the Windows audio-device name, run:
            Enter the following command in command prompt or PowerShell.
            @code{}
            %ProgramFiles%\Apollo\tools\audio-info.exe
            @endcode
            If you have multiple audio devices with identical names, use the Device ID instead.
            }
            @attention{If you want to mute the host speakers, use
            [virtual_sink](#virtual_sink) instead.}
        </td>
    </tr>
    <tr>
        <td>Default</td>
        <td colspan="2">Apollo will select the default Windows audio device.</td>
    </tr>
    <tr>
        <td>Example (Windows)</td>
        <td colspan="2">@code{}
            audio_sink = Speakers (High Definition Audio Device)
            @endcode</td>
    </tr>
</table>

### virtual_sink

<table>
    <tr>
        <td>Description</td>
        <td colspan="2">
            The virtual Windows audio device, such as Steam Streaming Speakers. This allows Apollo to stream audio,
            while muting the speakers.
            @tip{See [audio_sink](#audio_sink)!}
            @tip{These are some options for virtual sound devices.
            * Steam Streaming Speakers
              * Steam must be installed.
              * Apollo installs the driver on Windows when Steam is available; otherwise use
                Steam Remote Play once to install it.
            * [Virtual Audio Cable](https://vb-audio.com/Cable)
            }
        </td>
    </tr>
    <tr>
        <td>Default</td>
        <td colspan="2">n/a</td>
    </tr>
    <tr>
        <td>Example</td>
        <td colspan="2">@code{}
            virtual_sink = Steam Streaming Speakers
            @endcode</td>
    </tr>
</table>

### adapter_name

<table>
    <tr>
        <td>Description</td>
        <td colspan="2">
            Select the video card you want to stream.
            @tip{To find the appropriate Windows value, follow these instructions.
            Enter the following command in command prompt or PowerShell.
            @code{}
            %ProgramFiles%\Apollo\tools\dxgi-info.exe
            @endcode
            For hybrid graphics systems, DXGI reports the outputs are connected to whichever graphics
            adapter that the application is configured to use, so it's not a reliable indicator of how the
            display is physically connected.
            }
        </td>
    </tr>
    <tr>
        <td>Default</td>
        <td colspan="2">Apollo will select the default NVIDIA video card.</td>
    </tr>
    <tr>
        <td>Example (Windows)</td>
        <td colspan="2">@code{}
            adapter_name = NVIDIA GeForce RTX 5080
            @endcode</td>
    </tr>
</table>

### output_name

<table>
    <tr>
        <td>Description</td>
        <td colspan="2">
            Select the Windows display device to stream.
            @tip{During Apollo startup, the log lists the available display devices:
            @code{}
            Info: Currently available display devices:
            [
              {
                "device_id": "{64243705-4020-5895-b923-adc862c3457e}",
                "display_name": "",
                "friendly_name": "IDD HDR",
                "info": null
              },
              {
                "device_id": "{77f67f3e-754f-5d31-af64-ee037e18100a}",
                "display_name": "",
                "friendly_name": "Apollo Virtual Display",
                "info": null
              },
              {
                "device_id": "{daeac860-f4db-5208-b1f5-cf59444fb768}",
                "display_name": "\\\\.\\DISPLAY1",
                "friendly_name": "ROG PG279Q",
                "info": {
                  "hdr_state": null,
                  "origin_point": {
                    "x": 0,
                    "y": 0
                  },
                  "primary": true,
                  "refresh_rate": {
                    "type": "rational",
                    "value": {
                      "denominator": 1000,
                      "numerator": 119998
                    }
                  },
                  "resolution": {
                    "height": 1440,
                    "width": 2560
                  },
                  "resolution_scale": {
                    "type": "rational",
                    "value": {
                      "denominator": 100,
                      "numerator": 100
                    }
                  }
                }
              }
            ]
            @endcode
            You need to use the `device_id` value.
            }
        </td>
    </tr>
    <tr>
        <td>Default</td>
        <td colspan="2">Apollo will select the default display.</td>
    </tr>
    <tr>
        <td>Example</td>
        <td colspan="2">@code{}
            output_name = {daeac860-f4db-5208-b1f5-cf59444fb768}
            @endcode</td>
    </tr>
</table>

### max_bitrate

<table>
    <tr>
        <td>Description</td>
        <td colspan="2">
            The maximum bitrate (in Kbps) that Apollo will encode the stream at. If set to 0, it uses the bitrate requested by Artemis.
        </td>
    </tr>
    <tr>
        <td>Default</td>
        <td colspan="2">@code{}
            0
            @endcode</td>
    </tr>
    <tr>
        <td>Example</td>
        <td colspan="2">@code{}
            max_bitrate = 5000
            @endcode</td>
    </tr>
</table>

### minimum_fps_target

<table>
    <tr>
        <td>Description</td>
        <td colspan="2">
            Apollo saves bandwidth when content on screen is static or has a low frame rate. Because many clients expect a constant stream of video frames, duplicate frames are sent when needed. This setting controls the lowest effective frame rate a stream can reach.
        </td>
    </tr>
    <tr>
        <td>Default</td>
        <td colspan="2">@code{}
            0
            @endcode</td>
    </tr>
    <tr>
        <td rowspan="3">Choices</td>
        <td>0</td>
        <td>Use half the stream's FPS as the minimum target.</td>
    </tr>
    <tr>
        <td>1-1000</td>
        <td>Specify your own value. The real minimum may differ from this value.</td>
    </tr>
</table>

## Network

### address_family

<table>
    <tr>
        <td>Description</td>
        <td colspan="2">
            Set the address family that Apollo will use.
        </td>
    </tr>
    <tr>
        <td>Default</td>
        <td colspan="2">@code{}
            ipv4
            @endcode</td>
    </tr>
    <tr>
        <td>Example</td>
        <td colspan="2">@code{}
            address_family = both
            @endcode</td>
    </tr>
    <tr>
        <td rowspan="2">Choices</td>
        <td>ipv4</td>
        <td>IPv4 only</td>
    </tr>
    <tr>
        <td>both</td>
        <td>IPv4+IPv6</td>
    </tr>
</table>

### bind_address

<table>
    <tr>
        <td>Description</td>
        <td colspan="2">
            Optionally bind Apollo's HTTP, RTSP, control, audio, and video services to one local IP
            address. This can prevent traffic from using an unintended interface on hosts with
            multiple Ethernet, Wi-Fi, or VPN adapters. Leave empty to listen on all interfaces.
            The address must be IPv4 when <code>address_family = ipv4</code>, or IPv6 when
            <code>address_family = both</code>. An invalid explicit address prevents the network
            services from starting; Apollo never falls back to all interfaces in that case.
        </td>
    </tr>
    <tr>
        <td>Default</td>
        <td colspan="2">Empty (all interfaces)</td>
    </tr>
    <tr>
        <td>Example</td>
        <td colspan="2">@code{}
            bind_address = 192.168.1.100
            @endcode</td>
    </tr>
</table>

### port

<table>
    <tr>
        <td>Description</td>
        <td colspan="2">
            Set the family of ports used by Apollo.
            Changing this value will offset other ports as shown in config UI.
        </td>
    </tr>
    <tr>
        <td>Default</td>
        <td colspan="2">@code{}
            47989
            @endcode</td>
    </tr>
    <tr>
        <td>Range</td>
        <td colspan="2">1029-65514</td>
    </tr>
    <tr>
        <td>Example</td>
        <td colspan="2">@code{}
            port = 47989
            @endcode</td>
    </tr>
</table>

### origin_web_ui_allowed

<table>
    <tr>
        <td>Description</td>
        <td colspan="2">
            The origin of the remote endpoint address that is not denied for HTTPS Web UI.
        </td>
    </tr>
    <tr>
        <td>Default</td>
        <td colspan="2">@code{}
            lan
            @endcode</td>
    </tr>
    <tr>
        <td>Example</td>
        <td colspan="2">@code{}
            origin_web_ui_allowed = lan
            @endcode</td>
    </tr>
    <tr>
        <td rowspan="3">Choices</td>
        <td>pc</td>
        <td>Only localhost may access the web ui</td>
    </tr>
    <tr>
        <td>lan</td>
        <td>Only LAN devices may access the web ui</td>
    </tr>
    <tr>
        <td>wan</td>
        <td>Anyone may access the web ui</td>
    </tr>
</table>

### ping_timeout

<table>
    <tr>
        <td>Description</td>
        <td colspan="2">
            How long to wait, in milliseconds, for data from Artemis before shutting down the stream.
        </td>
    </tr>
    <tr>
        <td>Default</td>
        <td colspan="2">@code{}
            10000
            @endcode</td>
    </tr>
    <tr>
        <td>Example</td>
        <td colspan="2">@code{}
            ping_timeout = 10000
            @endcode</td>
    </tr>
</table>

### session_resume_grace

<table>
    <tr>
        <td>Description</td>
        <td colspan="2">
            How long, in milliseconds, to retain a launched app, virtual display, and host streaming setup after the remote client disconnects. Apollo permits one active remote stream; another launch is rejected until the current stream has fully stopped. A reconnect during this window resumes the same desktop without rebuilding it or losing its windows. When the grace expires, Apollo terminates the retained remote app. Set to 0 to terminate immediately unless the accepted session is still completing its RTSP/control handshake; Apollo preserves that handoff for at least @code{ping_timeout}. Valid range: 0 to 600000 (10 minutes).
        </td>
    </tr>
    <tr>
        <td>Default</td>
        <td colspan="2">@code{}
            60000
            @endcode</td>
    </tr>
    <tr>
        <td>Example</td>
        <td colspan="2">@code{}
            session_resume_grace = 60000
            @endcode</td>
    </tr>
</table>

### packetsize

<table>
    <tr>
        <td>Description</td>
        <td colspan="2">
            Optional ceiling for the video packet size requested by the client. Lowering it can avoid
            fragmentation and micro-stutter on low-MTU Wi-Fi or VPN links. Account for tunnel, IP, UDP,
            RTP, and encryption overhead when choosing a value. Smaller packets increase packet and FEC
            overhead, so a lower bitrate may be required.
        </td>
    </tr>
    <tr>
        <td>Default</td>
        <td colspan="2">@code{}
            0
            @endcode</td>
    </tr>
    <tr>
        <td>Range</td>
        <td colspan="2">0 (disabled), or 200-65459</td>
    </tr>
    <tr>
        <td>Example</td>
        <td colspan="2">@code{}
            packetsize = 1346
            @endcode</td>
    </tr>
</table>

## Config Files

### file_apps

<table>
    <tr>
        <td>Description</td>
        <td colspan="2">
            The application configuration file path. The file contains a JSON formatted list of applications that
            can be started by Artemis.
        </td>
    </tr>
    <tr>
        <td>Default</td>
        <td colspan="2">@code{}
            apps.json
            @endcode</td>
    </tr>
    <tr>
        <td>Example</td>
        <td colspan="2">@code{}
            file_apps = apps.json
            @endcode</td>
    </tr>
</table>

### credentials_file

<table>
    <tr>
        <td>Description</td>
        <td colspan="2">
            The file where user credentials for the UI are stored.
        </td>
    </tr>
    <tr>
        <td>Default</td>
        <td colspan="2">@code{}
            sunshine_state.json
            @endcode</td>
    </tr>
    <tr>
        <td>Example</td>
        <td colspan="2">@code{}
            credentials_file = sunshine_state.json
            @endcode</td>
    </tr>
</table>

### log_path

<table>
    <tr>
        <td>Description</td>
        <td colspan="2">
            The path where the Apollo log is stored.
        </td>
    </tr>
    <tr>
        <td>Default</td>
        <td colspan="2">@code{}
            sunshine.log
            @endcode</td>
    </tr>
    <tr>
        <td>Example</td>
        <td colspan="2">@code{}
            log_path = sunshine.log
            @endcode</td>
    </tr>
</table>

### pkey

<table>
    <tr>
        <td>Description</td>
        <td colspan="2">
            The private key used for the web UI and Artemis client pairing. For best compatibility,
            this should be an RSA-2048 private key.
            @warning{Not all Artemis clients support ECDSA keys or RSA key lengths other than 2048 bits.}
        </td>
    </tr>
    <tr>
        <td>Default</td>
        <td colspan="2">@code{}
            credentials/cakey.pem
            @endcode</td>
    </tr>
    <tr>
        <td>Example</td>
        <td colspan="2">@code{}
            pkey = /dir/pkey.pem
            @endcode</td>
    </tr>
</table>

### cert

<table>
    <tr>
        <td>Description</td>
        <td colspan="2">
            The certificate used for the web UI and Artemis client pairing. For best compatibility,
            this should have an RSA-2048 public key.
            @warning{Not all Artemis clients support ECDSA keys or RSA key lengths other than 2048 bits.}
        </td>
    </tr>
    <tr>
        <td>Default</td>
        <td colspan="2">@code{}
            credentials/cacert.pem
            @endcode</td>
    </tr>
    <tr>
        <td>Example</td>
        <td colspan="2">@code{}
            cert = /dir/cert.pem
            @endcode</td>
    </tr>
</table>

### file_state

<table>
    <tr>
        <td>Description</td>
        <td colspan="2">
            The file where Apollo's current state is stored.
        </td>
    </tr>
    <tr>
        <td>Default</td>
        <td colspan="2">@code{}
            sunshine_state.json
            @endcode</td>
    </tr>
    <tr>
        <td>Example</td>
        <td colspan="2">@code{}
            file_state = sunshine_state.json
            @endcode</td>
    </tr>
</table>

## Advanced

### fec_percentage

<table>
    <tr>
        <td>Description</td>
        <td colspan="2">
            Percentage of error correcting packets per data packet in each video frame.
            @warning{Higher values can correct for more network packet loss,
            but at the cost of increasing bandwidth usage.}
        </td>
    </tr>
    <tr>
        <td>Default</td>
        <td colspan="2">@code{}
            20
            @endcode</td>
    </tr>
    <tr>
        <td>Range</td>
        <td colspan="2">1-255</td>
    </tr>
    <tr>
        <td>Example</td>
        <td colspan="2">@code{}
            fec_percentage = 20
            @endcode</td>
    </tr>
</table>

### sbs_3d_profile

<table>
    <tr><td>Description</td><td>Select the startup Host SBS parameter profile. Explicit top-level <code>sbs_3d_*</code> settings override the selected profile.</td></tr>
    <tr><td>Default</td><td><code>apollo</code></td></tr>
</table>

## NVIDIA NVENC Encoder

### nvenc_preset

<table>
    <tr>
        <td>Description</td>
        <td colspan="2">
            NVENC encoder performance preset.
            Higher numbers improve compression (quality at given bitrate) at the cost of increased encoding latency.
            Recommended to change only when limited by network or decoder, otherwise similar effect can be accomplished
            by increasing bitrate.
            This option applies to Apollo's native NVENC path.
        </td>
    </tr>
    <tr>
        <td>Default</td>
        <td colspan="2">@code{}
            1
            @endcode</td>
    </tr>
    <tr>
        <td>Example</td>
        <td colspan="2">@code{}
            nvenc_preset = 1
            @endcode</td>
    </tr>
    <tr>
        <td rowspan="7">Choices</td>
        <td>1</td>
        <td>P1 (fastest)</td>
    </tr>
    <tr>
        <td>2</td>
        <td>P2</td>
    </tr>
    <tr>
        <td>3</td>
        <td>P3</td>
    </tr>
    <tr>
        <td>4</td>
        <td>P4</td>
    </tr>
    <tr>
        <td>5</td>
        <td>P5</td>
    </tr>
    <tr>
        <td>6</td>
        <td>P6</td>
    </tr>
    <tr>
        <td>7</td>
        <td>P7 (slowest)</td>
    </tr>
</table>

### nvenc_twopass

<table>
    <tr>
        <td>Description</td>
        <td colspan="2">
            Enable two-pass mode in NVENC encoder.
            This allows to detect more motion vectors, better distribute bitrate across the frame and more strictly
            adhere to bitrate limits. Disabling it is not recommended since this can lead to occasional bitrate
            overshoot and subsequent packet loss.
            This option applies to Apollo's native NVENC path.
        </td>
    </tr>
    <tr>
        <td>Default</td>
        <td colspan="2">@code{}
            quarter_res
            @endcode</td>
    </tr>
    <tr>
        <td>Example</td>
        <td colspan="2">@code{}
            nvenc_twopass = quarter_res
            @endcode</td>
    </tr>
    <tr>
        <td rowspan="3">Choices</td>
        <td>disabled</td>
        <td>One pass (fastest)</td>
    </tr>
    <tr>
        <td>quarter_res</td>
        <td>Two passes, first pass at quarter resolution (faster)</td>
    </tr>
    <tr>
        <td>full_res</td>
        <td>Two passes, first pass at full resolution (slower)</td>
    </tr>
</table>

### nvenc_spatial_aq

<table>
    <tr>
        <td>Description</td>
        <td colspan="2">
            Assign higher QP values to flat regions of the video.
            Recommended to enable when streaming at lower bitrates.
            This option applies to Apollo's native NVENC path.
            @warning{Enabling this option may reduce performance.}
        </td>
    </tr>
    <tr>
        <td>Default</td>
        <td colspan="2">@code{}
            disabled
            @endcode</td>
    </tr>
    <tr>
        <td>Example</td>
        <td colspan="2">@code{}
            nvenc_spatial_aq = disabled
            @endcode</td>
    </tr>
</table>

### nvenc_hevc_unidirectional_b

<table>
    <tr>
        <td>Description</td>
        <td colspan="2">
            Replace HEVC P pictures with unidirectional B pictures whose reference lists contain only past pictures.
            This can improve compression quality without the reordering delay of conventional B pictures.
            The option is ignored for H.264 and AV1 and falls back to ordinary P pictures when unsupported by the GPU.
            This option applies to HEVC on Apollo's native NVENC path.
        </td>
    </tr>
    <tr>
        <td>Default</td>
        <td colspan="2">@code{}
            disabled
            @endcode</td>
    </tr>
    <tr>
        <td>Example</td>
        <td colspan="2">@code{}
            nvenc_hevc_unidirectional_b = enabled
            @endcode</td>
    </tr>
</table>

### nvenc_vbv_increase

<table>
    <tr>
        <td>Description</td>
        <td colspan="2">
            Single-frame VBV/HRD percentage increase.
            By default Apollo uses single-frame VBV/HRD, which means any encoded video frame size is not expected to
            exceed requested bitrate divided by requested frame rate. Relaxing this restriction can be beneficial and
            act as low-latency variable bitrate, but may also lead to packet loss if the network doesn't have buffer
            headroom to handle bitrate spikes. Maximum accepted value is 400, which corresponds to 5x increased
            encoded video frame upper size limit.
            This option applies to Apollo's native NVENC path.
            @warning{Can lead to network packet loss.}
        </td>
    </tr>
    <tr>
        <td>Default</td>
        <td colspan="2">@code{}
            0
            @endcode</td>
    </tr>
    <tr>
        <td>Range</td>
        <td colspan="2">0-400</td>
    </tr>
    <tr>
        <td>Example</td>
        <td colspan="2">@code{}
            nvenc_vbv_increase = 0
            @endcode</td>
    </tr>
</table>

### nvenc_realtime_hags

<table>
    <tr>
        <td>Description</td>
        <td colspan="2">
            Use realtime gpu scheduling priority in NVENC when hardware accelerated gpu scheduling (HAGS) is enabled
            in Windows. Currently, NVIDIA drivers may freeze in encoder when HAGS is enabled, realtime priority is used
            and VRAM utilization is close to maximum. Disabling this option lowers the priority to high, sidestepping
            the freeze at the cost of reduced capture performance when the GPU is heavily loaded.
            This option applies to Apollo's native NVENC path.
        </td>
    </tr>
    <tr>
        <td>Default</td>
        <td colspan="2">@code{}
            enabled
            @endcode</td>
    </tr>
    <tr>
        <td>Example</td>
        <td colspan="2">@code{}
            nvenc_realtime_hags = enabled
            @endcode</td>
    </tr>
</table>

### nvenc_latency_over_power

<table>
    <tr>
        <td>Description</td>
        <td colspan="2">
            Adaptive P-State algorithm which NVIDIA drivers employ doesn't work well with low latency streaming,
            so Apollo requests high power mode explicitly.
            This option applies to Apollo's native NVENC path.
            @warning{Disabling this is not recommended since this can lead to significantly increased encoding latency.}
            @note{Applies to Windows only.}
        </td>
    </tr>
    <tr>
        <td>Default</td>
        <td colspan="2">@code{}
            enabled
            @endcode</td>
    </tr>
    <tr>
        <td>Example</td>
        <td colspan="2">@code{}
            nvenc_latency_over_power = enabled
            @endcode</td>
    </tr>
</table>

### nvenc_opengl_vulkan_on_dxgi

<table>
    <tr>
        <td>Description</td>
        <td colspan="2">
            Apollo can't capture fullscreen OpenGL and Vulkan programs at full frame rate unless they present on
            top of DXGI. With this option enabled Apollo changes the global Vulkan/OpenGL present method to
            "Prefer layered on DXGI Swapchain". This system-wide setting is reverted when Apollo exits.
            This option applies to Apollo's native NVENC path.
        </td>
    </tr>
    <tr>
        <td>Default</td>
        <td colspan="2">@code{}
            enabled
            @endcode</td>
    </tr>
    <tr>
        <td>Example</td>
        <td colspan="2">@code{}
            nvenc_opengl_vulkan_on_dxgi = enabled
            @endcode</td>
    </tr>
</table>

<div class="section_buttons">

| Previous          |                            Next |
|:------------------|--------------------------------:|
| [Legal](legal.md) | [App Examples](app_examples.md) |

</div>

<details style="display: none;">
  <summary></summary>
  [TOC]
</details>
