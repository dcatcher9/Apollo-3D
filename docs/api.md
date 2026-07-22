# API

Apollo has a RESTful configuration API which is used by the Web UI.

Unless otherwise specified, authentication is required for all API calls. Authenticate with
`POST /api/login`, then send the returned `auth` cookie on later requests. Unsafe methods also
require browser source metadata that exactly matches the HTTPS request host and port. The examples
below include the required `Origin` header for non-browser clients. Reverse proxies must preserve
the original `Host` value.

@htmlonly
<script src="api.js"></script>
@endhtmlonly

## GET /api/apps
@copydoc confighttp::getApps()

## POST /api/apps
@copydoc confighttp::saveApp()

## POST /api/apps/close
@copydoc confighttp::closeApp()

## POST /api/apps/delete
@copydoc confighttp::deleteApp()

## POST /api/apps/reorder
@copydoc confighttp::reorderApps()

## GET /api/clients/list
@copydoc confighttp::getClients()

## POST /api/clients/unpair
@copydoc confighttp::unpair()

## POST /api/clients/unpair-all
@copydoc confighttp::unpairAll()

## POST /api/clients/update
@copydoc confighttp::updateClient()

## POST /api/clients/disconnect
@copydoc confighttp::disconnect()

## GET /api/ar-glasses
@copydoc confighttp::getArGlassDevices()

## POST /api/ar-glasses
@copydoc confighttp::setArGlassDevice()

## GET /api/config
@copydoc confighttp::getConfig()

## GET /api/configLocale
@copydoc confighttp::getLocale()

## POST /api/config
@copydoc confighttp::saveConfig()

## POST /api/covers/upload
@copydoc confighttp::uploadCover()

## GET /api/logs
@copydoc confighttp::getLogs()

## POST /api/password
@copydoc confighttp::savePassword()

## POST /api/pin
@copydoc confighttp::savePin()

## POST /api/restart
@copydoc confighttp::restart()

## POST /api/quit
@copydoc confighttp::quit()

<div class="section_buttons">

| Previous                                    |                                  Next |
|:--------------------------------------------|--------------------------------------:|
| [Performance Tuning](performance_tuning.md) | [Troubleshooting](troubleshooting.md) |

</div>

<details style="display: none;">
  <summary></summary>
  [TOC]
</details>
