# Web API

Sunshine 3D's Web UI uses an HTTPS JSON API. It is an administration interface for the Windows
host, not a public remote-control protocol. Keep it on the host or a trusted local network.

Requests from this PC or a network allowed by `origin_web_ui_allowed` use the local trust boundary
and do not require a sign-in. If WAN Web UI access is explicitly enabled, authenticate with
`POST /api/login` and retain the returned `auth` cookie. State-changing requests must also carry an
`Origin` header whose HTTPS host and port match the request. A reverse proxy must preserve `Host`.

The API is consumed by the bundled Web UI and may evolve with it. Integrations should tolerate new
response fields and should never write configuration files behind Sunshine 3D's back.

@htmlonly
<script src="api.js"></script>
@endhtmlonly

## Pairing and authentication

| Method | Route | Purpose |
|---|---|---|
| `POST` | `/api/login` | Authenticate WAN Web UI access |
| `POST` | `/api/pin` | Complete primary four-digit PIN pairing |
| `POST` | `/api/otp` | Create the secondary QR/one-time pairing payload |
| `POST` | `/api/password` | Change Web UI credentials |

## Applications and host control

| Method | Route | Purpose |
|---|---|---|
| `GET`, `POST` | `/api/apps` | List or save applications |
| `POST` | `/api/apps/close` | Close the active application |
| `POST` | `/api/apps/delete` | Delete an application entry |
| `POST` | `/api/apps/reorder` | Reorder applications |
| `POST` | `/api/covers/upload` | Upload or fetch an application cover |
| `GET` | `/api/logs` | Read the current host log |
| `POST` | `/api/restart` | Restart Sunshine 3D |
| `POST` | `/api/quit` | Stop Sunshine 3D |

## Devices and presentation

| Method | Route | Purpose |
|---|---|---|
| `GET` | `/api/clients/list` | List paired and active clients |
| `POST` | `/api/clients/update` | Change a client's permissions |
| `POST` | `/api/clients/disconnect` | Disconnect the active client |
| `POST` | `/api/clients/unpair` | Remove one paired client |
| `POST` | `/api/clients/unpair-all` | Remove every paired client |
| `GET`, `POST` | `/api/ar-glasses` | Inspect or select the Windows local-AR display |

## Configuration

| Method | Route | Purpose |
|---|---|---|
| `GET`, `POST` | `/api/config` | Read or save host configuration |
| `GET` | `/api/configLocale` | Read locale metadata |

The field-level contract is documented in [Configuration](configuration.md).

## Offline Host 3D conversion

| Method | Route | Purpose |
|---|---|---|
| `GET` | `/api/offline-sbs/capabilities` | Report whether the native worker and codecs are available |
| `GET`, `POST` | `/api/offline-sbs/jobs` | List jobs or create a conversion |
| `GET` | `/api/offline-sbs/jobs/{id}` | Read one job |
| `POST` | `/api/offline-sbs/jobs/{id}/cancel` | Cancel one job |
| `GET` | `/api/offline-sbs/jobs/{id}/scene-audit` | Download its bounded scene audit |

The job schema, path restrictions, media behavior, and GPU lease are owned by
[Offline Host 3D conversion](whole-clip-sbs-pipeline.md). Do not infer them from this route index.

## Generated examples

The following endpoint sections are connected to the source-level Doxygen examples.

### GET /api/apps
@copydoc confighttp::getApps()

### POST /api/apps
@copydoc confighttp::saveApp()

### GET /api/clients/list
@copydoc confighttp::getClients()

### POST /api/config
@copydoc confighttp::saveConfig()

### POST /api/pin
@copydoc confighttp::savePin()
