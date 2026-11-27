<template>
  <Navbar></Navbar>
  <main id="content" class="container devices-page">
    <header class="devices-page-header">
      <div>
        <div class="devices-eyebrow">Sunshine 3D</div>
        <h1 class="devices-title">{{ $t('pin.devices_title') }}</h1>
        <p class="devices-subtitle">{{ $t('pin.devices_subtitle') }}</p>
      </div>
    </header>

    <section class="surface-card pair-card" aria-labelledby="pair-artemis-heading">
      <header class="pair-intro">
        <div class="pair-intro-copy">
          <h2 id="pair-artemis-heading">{{ $t('pin.pair_device') }}</h2>
          <p>{{ $t('pin.pair_device_desc') }}</p>
        </div>
        <div v-if="pairedDeviceName" class="alert alert-success mb-0" role="status">
          <i class="fa-solid fa-circle-check me-2"></i>
          {{ $t('pin.device_ready', { name: pairedDeviceName }) }}
        </div>
      </header>

      <div class="pair-method-grid">
        <article id="PIN" class="pair-method-card is-primary">
          <header class="pair-method-header">
            <div class="pair-icon"><i class="fa-solid fa-key"></i></div>
            <div>
              <h3>{{ $t('pin.enter_pin') }}</h3>
              <p>{{ $t('pin.enter_pin_desc') }}</p>
            </div>
            <span class="pair-method-step">{{ $t('pin.primary_method') }}</span>
          </header>
          <form class="pin-pairing-form" @submit.prevent="pairWithPin">
            <div>
              <label for="pairing-pin" class="form-label">{{ $t('pin.pairing_pin') }}</label>
              <input id="pairing-pin" type="text" inputmode="numeric" pattern="\d*" autocomplete="one-time-code"
                placeholder="0000" v-model.trim="pairingPin" class="form-control pin-code-input" required />
            </div>
            <div>
              <label for="pairing-device-name" class="form-label">{{ $t('pin.device_name') }}</label>
              <input id="pairing-device-name" type="text" :placeholder="$t('pin.device_name')"
                v-model.trim="pairingDeviceName" class="form-control pin-device-name" />
              <div class="form-text">{{ $t('pin.device_name_note') }}</div>
            </div>
            <button class="btn btn-apollo pair-method-action" :disabled="pinPairingPending">
              <span v-if="pinPairingPending" class="spinner-border spinner-border-sm" aria-hidden="true"></span>
              <i v-else class="fa-solid fa-link"></i>
              <span>{{ pinPairingPending ? $t('pin.pairing_pending') : $t('pin.send') }}</span>
            </button>
          </form>
          <div v-if="pinMessage" class="alert pairing-feedback mb-0" :class="'alert-' + pinStatus" role="status">
            {{ pinMessage }}
          </div>
        </article>

        <article class="pair-method-card qr-method-card" :class="{ 'is-active': pairingActive }">
          <div v-if="!pairingActive" class="qr-method-idle">
            <div class="pair-copy">
              <div class="pair-icon"><i class="fa-solid fa-qrcode"></i></div>
              <h3>{{ $t('pin.scan_qr') }}</h3>
              <p>{{ $t('pin.scan_qr_desc') }}</p>
              <div v-if="otpMessage" class="alert mb-0"
                :class="[pairedDeviceName ? 'mt-2' : '', 'alert-' + otpStatus]" role="status">
                {{ otpMessage }}
              </div>
              <div v-if="!pairedDeviceName && !otpMessage" class="pair-security-note">
                <i class="fa-solid fa-shield-halved"></i>
                {{ $t('pin.qr_single_use') }}
              </div>
            </div>
            <button type="button" class="btn btn-apollo-quiet pair-method-action" :disabled="pairingPending"
              @click="requestOTP">
              <span v-if="pairingPending" class="spinner-border spinner-border-sm" aria-hidden="true"></span>
              <i v-else class="fa-solid fa-mobile-screen-button"></i>
              {{ pairingPending ? $t('pin.qr_preparing') : $t('pin.show_qr') }}
            </button>
          </div>

          <div v-else class="pairing-layout">
            <div id="qrRef" class="pairing-qr" :aria-label="$t('pin.qr_aria')"></div>
            <div>
              <div class="pairing-state"><span class="pairing-state-dot"></span> {{ $t('pin.qr_open_remaining', { time: pairingTimeLabel }) }}</div>
              <h3>{{ $t('pin.scan_with_moonlight') }}</h3>
              <p>{{ $t('pin.scan_with_moonlight_desc') }}</p>
              <div class="pairing-actions">
                <a class="btn btn-apollo" :href="deepLink">
                  <i class="fa-solid fa-arrow-up-right-from-square"></i>
                  {{ $t('pin.open_in_moonlight') }}
                </a>
                <button type="button" class="btn btn-outline-secondary" @click="requestOTP">
                  <i class="fa-solid fa-rotate"></i>
                  {{ $t('pin.new_code') }}
                </button>
              </div>
              <div v-if="otpMessage" class="alert mt-3 mb-0" :class="'alert-' + otpStatus" role="status">
                {{ otpMessage }}
              </div>
            </div>
          </div>
        </article>
      </div>

    </section>

    <section aria-labelledby="remote-devices-heading">
      <div class="section-heading">
        <div>
          <h2 id="remote-devices-heading">{{ $t('pin.remote_devices') }}</h2>
          <p>{{ $t('pin.remote_devices_desc') }}</p>
        </div>
      </div>

      <div v-if="clientsError" class="alert alert-danger">{{ clientsError }}</div>
      <div v-if="showApplyMessage" class="alert alert-success d-flex align-items-center" role="status">
        <span><i class="fa-solid fa-circle-check me-2"></i>{{ $t('pin.device_removed') }}</span>
        <button type="button" class="btn-close ms-auto" :aria-label="$t('pin.dismiss')" @click="showApplyMessage = false"></button>
      </div>

      <div class="device-grid">
        <article v-for="client in clients" :key="client.uuid" class="surface-card device-card" :class="{'is-editing': client.editing}">
          <template v-if="!client.editing">
            <div class="device-card-header">
              <div class="device-icon"><i class="fa-solid fa-headset"></i></div>
              <div class="min-w-0">
                <h3 class="device-name">{{ displayClientName(client) }}</h3>
                <div class="device-state">
                  <span class="status-dot" :class="{offline: !client.connected}"></span>
                  {{ client.connected ? $t('pin.state_connected') : $t('pin.state_ready') }}
                </div>
              </div>
              <div class="device-card-actions">
                <button v-if="client.connected" type="button" class="btn btn-outline-warning icon-button" :title="$t('pin.disconnect')" :aria-label="$t('pin.disconnect_device', { name: displayClientName(client) })" @click="disconnectClient(client.uuid)">
                  <i class="fa-solid fa-link-slash"></i>
                </button>
                <button type="button" class="btn btn-outline-secondary icon-button" :title="$t('pin.device_settings')" :aria-label="$t('pin.device_settings_for', { name: displayClientName(client) })" @click="editClient(client)">
                  <i class="fa-solid fa-gear"></i>
                </button>
                <button type="button" class="btn btn-outline-danger icon-button" :title="$t('pin.remove_device')" :aria-label="$t('pin.remove_named', { name: displayClientName(client) })" @click="unpairSingle(client)">
                  <i class="fa-solid fa-trash-can"></i>
                </button>
              </div>
            </div>

            <div class="device-facts">
              <div class="device-fact">
                <span class="device-fact-label">{{ $t('pin.access') }}</span>
                <span class="device-fact-value">
                  {{ accessLabel(client.perm) }}
                </span>
              </div>
              <div class="device-fact">
                <span class="device-fact-label">{{ $t('pin.clipboard_sharing') }}</span>
                <span class="device-fact-value">{{ hasClipboard(client.perm) ? $t('pin.on') : $t('pin.off') }}</span>
              </div>
            </div>
          </template>

          <form v-else @submit.prevent="saveClient(client)">
            <div class="editor-heading">
              <h3>{{ $t('pin.device_settings') }}</h3>
              <button type="button" class="btn-close" :aria-label="$t('pin.close_device_settings')" @click="cancelEdit(client)"></button>
            </div>

            <div class="mb-4">
              <label class="form-label fw-semibold" :for="`device-name-${client.uuid}`">{{ $t('pin.device_name_label') }}</label>
              <input :id="`device-name-${client.uuid}`" v-model.trim="client.editName" class="form-control" type="text" :placeholder="$t('pin.device_name')">
            </div>

            <div class="editor-grid">
              <section class="editor-section">
                <h4>{{ $t('pin.access') }}</h4>
                <div class="access-options">
                  <button type="button" class="access-option" :class="{active: client.editAccessPreset === 'stream_control'}" @click="setAccessPreset(client, 'stream_control')">
                    <strong>{{ $t('pin.access_stream_control') }}</strong>
                    <span>{{ $t('pin.access_stream_control_desc') }}</span>
                  </button>
                  <button type="button" class="access-option" :class="{active: client.editAccessPreset === 'view_only'}" @click="setAccessPreset(client, 'view_only')">
                    <strong>{{ $t('pin.access_view_only') }}</strong>
                    <span>{{ $t('pin.access_view_only_desc') }}</span>
                  </button>
                  <button type="button" class="access-option" :class="{active: client.editAccessPreset === 'custom'}" @click="setAccessPreset(client, 'custom')">
                    <strong>{{ $t('pin.access_custom') }}</strong>
                    <span>{{ $t('pin.access_custom_desc') }}</span>
                  </button>
                </div>
              </section>

              <section class="editor-section">
                <h4>{{ $t('pin.preferences') }}</h4>
                <div class="preference-list">
                  <div class="preference-row">
                    <div>
                      <label :for="`clipboard-${client.uuid}`">{{ $t('pin.clipboard_sharing') }}</label>
                      <small>{{ $t('pin.clipboard_sharing_desc') }}</small>
                    </div>
                    <div class="form-check form-switch m-0">
                      <input :id="`clipboard-${client.uuid}`" class="form-check-input" type="checkbox" role="switch" v-model="client.editClipboard" @change="setClipboard(client)">
                    </div>
                  </div>
                </div>
              </section>
            </div>

            <details class="advanced-disclosure" :open="client.showAdvanced">
              <summary>{{ $t('pin.advanced_settings') }}</summary>
              <div class="advanced-panel">
                <div>
                  <h4 class="h6 mb-3">{{ $t('pin.granular_permissions') }}</h4>
                  <div class="permission-groups">
                    <div v-for="group in permissionGroups" :key="group.name" class="permission-group">
                      <h5>{{ $t(group.labelKey) }}</h5>
                      <div v-for="perm in group.permissions" :key="perm.name" class="form-check">
                        <input
                          type="checkbox"
                          class="form-check-input"
                          :id="`${client.uuid}-${perm.name}`"
                          :checked="isSuppressed(client.editPerm, perm.name, perm.suppressed_by) || checkPermission(client.editPerm, perm.name)"
                          :disabled="isSuppressed(client.editPerm, perm.name, perm.suppressed_by)"
                          @change="togglePermission(client, perm.name)"
                        >
                        <label class="form-check-label" :for="`${client.uuid}-${perm.name}`">{{ $t(`permissions.${perm.name}`) }}</label>
                      </div>
                    </div>
                  </div>
                </div>

                <details>
                  <summary class="small text-secondary">{{ $t('pin.technical_details') }}</summary>
                  <div class="font-monospace small text-secondary mt-2">{{ $t('pin.device_id', { id: client.uuid }) }}</div>
                  <div class="font-monospace small text-secondary">{{ $t('pin.permission_code', { code: permToStr(client.editPerm) }) }}</div>
                </details>
              </div>
            </details>

            <div v-if="client.saveError" class="alert alert-danger mt-3 mb-0">{{ client.saveError }}</div>
            <div class="editor-actions">
              <button type="button" class="btn btn-outline-secondary" :disabled="client.saving" @click="cancelEdit(client)">{{ $t('_common.cancel') }}</button>
              <button type="submit" class="btn btn-apollo" :disabled="client.saving">
                <span v-if="client.saving" class="spinner-border spinner-border-sm" aria-hidden="true"></span>
                {{ client.saving ? $t('pin.saving') : $t('pin.save_device') }}
              </button>
            </div>
          </form>
        </article>

        <div v-if="clientsLoaded && clients.length === 0" class="surface-card empty-devices">
          <i class="fa-solid fa-headset"></i>
          <div>
            <strong>{{ $t('pin.no_devices') }}</strong>
            <div class="small mt-1">{{ $t('pin.no_devices_desc') }}</div>
          </div>
        </div>
      </div>

      <details v-if="clients.length > 0" class="support-disclosure">
        <summary>{{ $t('pin.device_support') }}</summary>
        <div class="support-disclosure-body">
          <div>
            <strong>{{ $t('pin.remove_all_title') }}</strong>
            <p>{{ $t('pin.remove_all_desc') }}</p>
          </div>
          <button type="button" class="btn btn-outline-danger" :disabled="unpairAllPressed" @click="unpairAll">
            {{ unpairAllPressed ? $t('pin.removing') : $t('pin.remove_all') }}
          </button>
        </div>
        <div v-if="unpairAllStatus === true" class="alert alert-success mt-3">{{ $t('pin.unpair_all_success') }}</div>
        <div v-if="unpairAllStatus === false" class="alert alert-danger mt-3">{{ $t('pin.unpair_all_error') }}</div>
      </details>
    </section>

    <ar-glasses class="ar-device-host" embedded></ar-glasses>
  </main>
</template>

<script>
import Navbar from './Navbar.vue'
import ArGlasses from './configs/tabs/ArGlasses.vue'
import { ApiError, getJson, postJson } from './api'

let pairingCountdownTimer = null;
let pairingPollTimer = null;

const qrContainer = document.createElement('div');
const qrCode = new QRCode(qrContainer);

const updateQR = (url) => {
  qrCode.clear();
  qrCode.makeCode(url);

  const refContainer = document.querySelector('#qrRef');
  if (refContainer) {
    refContainer.replaceChildren(qrContainer);
  }
};

const clearPairingTimers = () => {
  if (pairingCountdownTimer !== null) clearInterval(pairingCountdownTimer);
  if (pairingPollTimer !== null) clearInterval(pairingPollTimer);
  pairingCountdownTimer = null;
  pairingPollTimer = null;
};

const generatePairingSecret = () => {
  const alphabet = '23456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz';
  const bytes = new Uint8Array(18);
  window.crypto.getRandomValues(bytes);
  return Array.from(bytes, byte => alphabet[byte % alphabet.length]).join('');
};

/**
 * Permissions:
 * 0x00003F00 - controller, touch, pen, mouse, keyboard, and microphone input
 * 0x00030000 - clipboard operations
 * 0x07000000 - list, view, and launch actions
 */
const permissionMapping = {
  input_controller: 0x00000100,
  input_touch: 0x00000200,
  input_pen: 0x00000400,
  input_mouse: 0x00000800,
  input_kbd: 0x00001000,
  input_microphone: 0x00002000,
  _all_inputs: 0x00003F00,

  clipboard_set: 0x00010000,
  clipboard_read: 0x00020000,
  _all_operations: 0x00030000,
  _clipboard: 0x00030000,

  list: 0x01000000,
  view: 0x02000000,
  launch: 0x04000000,
  _allow_view: 0x06000000,
  _all_actions: 0x07000000,

  _default: 0x03000000,
  _all: 0x07033F00
};

const permissionGroups = [
  { name: 'Streaming', labelKey: 'pin.permission_group_streaming', permissions: [
    { name: 'list', suppressed_by: ['view', 'launch'] },
    { name: 'view', suppressed_by: ['launch'] },
    { name: 'launch', suppressed_by: [] }
  ] },
  { name: 'Clipboard', labelKey: 'pin.permission_group_clipboard', permissions: [
    { name: 'clipboard_set', suppressed_by: [] },
    { name: 'clipboard_read', suppressed_by: [] }
  ] },
  { name: 'Input', labelKey: 'pin.permission_group_input', permissions: [
    { name: 'input_controller', suppressed_by: [] },
    { name: 'input_touch', suppressed_by: [] },
    { name: 'input_pen', suppressed_by: [] },
    { name: 'input_mouse', suppressed_by: [] },
    { name: 'input_kbd', suppressed_by: [] },
    { name: 'input_microphone', suppressed_by: [] }
  ] }
];

let currentEditingClient = null;

const data = () => ({
  otp: '',
  passphrase: '',
  hostAddr: '',
  hostPort: '',
  hostName: '',
  otpMessage: '',
  otpStatus: 'info',
  pairingActive: false,
  pairingPending: false,
  pairingSecondsRemaining: 180,
  pairingInitialClientIds: [],
  pairedDeviceName: '',
  pinPairingPending: false,
  pairingPin: '',
  pairingDeviceName: '',
  pinMessage: '',
  pinStatus: 'info',
  permissionGroups,
  clients: [],
  clientsLoaded: false,
  clientsError: '',
  showApplyMessage: false,
  unpairAllPressed: false,
  unpairAllStatus: null
});

export default {
  components: {
    Navbar,
    ArGlasses
  },
  inject: ['i18n'],
  data,
  computed: {
    deepLink() {
      const params = new URLSearchParams({
        pin: this.otp,
        passphrase: this.passphrase,
        name: this.hostName
      });
      return `art://${this.hostAddr}:${this.hostPort}?${params.toString()}`;
    },
    pairingTimeLabel() {
      const minutes = Math.floor(this.pairingSecondsRemaining / 60);
      const seconds = this.pairingSecondsRemaining % 60;
      return `${minutes}:${seconds.toString().padStart(2, '0')}`;
    }
  },
  created() {
    this.refreshClients();
  },
  beforeUnmount() {
    clearPairingTimers();
  },
  methods: {
    displayClientName(client) {
      return client.name || this.i18n.t('pin.unpair_single_unknown');
    },
    async requestOTP() {
      if (this.pairingPending) return;

      this.pairingPending = true;
      this.pairedDeviceName = '';
      this.otpMessage = '';
      clearPairingTimers();

      try {
        if (!this.clientsLoaded) await this.refreshClients();
        this.pairingInitialClientIds = this.clients.map(client => client.uuid);
        this.passphrase = generatePairingSecret();

        const body = await postJson('./api/otp', {passphrase: this.passphrase, deviceName: ''});

        this.otp = body.otp;
        this.hostName = body.name;
        const isLocalHost = ['localhost', '127.0.0.1', '[::1]'].includes(location.hostname);
        this.hostAddr = isLocalHost ? body.ip : location.hostname;
        const managementPort = Number.parseInt(location.port, 10);
        this.hostPort = Number.isFinite(managementPort) ? managementPort - 1 : 47989;
        this.pairingSecondsRemaining = 180;
        this.pairingActive = true;
        this.otpStatus = 'info';

        await this.$nextTick();
        updateQR(this.deepLink);

        pairingCountdownTimer = setInterval(() => {
          this.pairingSecondsRemaining -= 1;
          if (this.pairingSecondsRemaining <= 0) this.expirePairing();
        }, 1000);
        pairingPollTimer = setInterval(() => this.refreshClients(true), 2000);
      } catch (error) {
        this.pairingActive = false;
        this.otpStatus = 'danger';
        this.otpMessage = error.message || String(error);
      } finally {
        this.pairingPending = false;
      }
    },
    expirePairing() {
      clearPairingTimers();
      this.pairingActive = false;
      this.otp = '';
      this.otpStatus = 'warning';
      this.otpMessage = this.i18n.t('pin.pairing_expired');
    },
    async completeArtemisPairing(client) {
      clearPairingTimers();
      this.pairingActive = false;
      this.otp = '';
      this.pairedDeviceName = this.displayClientName(client);
      this.otpMessage = '';

      try {
        await this.applyArtemisDefaults(client);
        await this.refreshClients();
      } catch (error) {
        this.otpStatus = 'warning';
        this.otpMessage = this.i18n.t('pin.defaults_not_saved', { error: error.message || error });
      }
    },
    async applyArtemisDefaults(client) {
      const payload = this.clientPayload(client);
      payload.perm = permissionMapping._all_actions | permissionMapping._all_inputs;

      await this.updateClient(payload);
    },
    async updateClient(payload) {
      // An unknown device is a successful reply whose status is false.
      const body = await postJson('./api/clients/update', payload);
      if (!body.status) throw new Error(body.error || body.message || this.i18n.t('pin.action_failed'));
    },
    async pairWithPin() {
      if (this.pinPairingPending) return;
      this.pinPairingPending = true;
      this.pinMessage = '';

      try {
        const body = await postJson('./api/pin', {pin: this.pairingPin, name: this.pairingDeviceName});
        if (!body.status) throw new Error(this.i18n.t('pin.pair_failure'));

        this.pinStatus = 'success';
        this.pinMessage = this.i18n.t('pin.pair_success');
        this.pairingPin = '';
        this.pairingDeviceName = '';
        setTimeout(() => this.refreshClients(), 1000);
      } catch (error) {
        this.pinStatus = 'danger';
        this.pinMessage = error instanceof ApiError ? this.i18n.t('pin.pair_failure') : error.message || String(error);
      } finally {
        this.pinPairingPending = false;
      }
    },
    editClient(client) {
      if (currentEditingClient) this.cancelEdit(currentEditingClient);

      client.editing = true;
      client.editPerm = client.perm;
      client.editName = client.name;
      client.editAccessPreset = this.accessPresetFor(client.perm);
      client.editClipboard = this.hasClipboard(client.perm);
      client.showAdvanced = client.editAccessPreset === 'custom';
      client.saveError = '';
      currentEditingClient = client;
    },
    cancelEdit(client) {
      if (currentEditingClient === client) currentEditingClient = null;
      client.editing = false;
      client.saveError = '';
    },
    clientPayload(client, overrides = {}) {
      const sourcePerm = client.editPerm === undefined ? client.perm : client.editPerm;
      return {
        uuid: client.uuid,
        name: client.editName === undefined ? client.name : client.editName,
        perm: sourcePerm & permissionMapping._all,
        ...overrides
      };
    },
    async saveClient(client) {
      client.saving = true;
      client.saveError = '';

      try {
        await this.updateClient(this.clientPayload(client));

        currentEditingClient = null;
        client.editing = false;
        await this.refreshClients();
      } catch (error) {
        client.saveError = this.i18n.t('pin.save_client_error') + (error.message || error);
      } finally {
        client.saving = false;
      }
    },
    accessPresetFor(perm) {
      const actions = perm & permissionMapping._all_actions;
      const inputs = perm & permissionMapping._all_inputs;
      if ((actions & permissionMapping.launch) && inputs === permissionMapping._all_inputs) return 'stream_control';
      if (!(actions & permissionMapping.launch) && (actions & permissionMapping._allow_view) && inputs === 0) return 'view_only';
      return 'custom';
    },
    accessLabel(perm) {
      // A client loaded without a permission word has none until the owner grants them.
      if ((perm & permissionMapping._all) === 0) return this.i18n.t('pin.access_none');
      return this.accessPresetLabel(this.accessPresetFor(perm));
    },
    accessPresetLabel(preset) {
      if (preset === 'stream_control') return this.i18n.t('pin.access_stream_control');
      if (preset === 'view_only') return this.i18n.t('pin.access_view_only');
      return this.i18n.t('pin.access_custom');
    },
    setAccessPreset(client, preset) {
      client.editAccessPreset = preset;
      if (preset === 'custom') {
        client.showAdvanced = true;
        return;
      }

      const clipboard = client.editPerm & permissionMapping._clipboard;
      client.editPerm &= ~(permissionMapping._all_actions | permissionMapping._all_inputs | permissionMapping._all_operations);
      client.editPerm |= clipboard;
      if (preset === 'stream_control') {
        client.editPerm |= permissionMapping._all_actions | permissionMapping._all_inputs;
      } else {
        client.editPerm |= permissionMapping._default;
      }
    },
    setClipboard(client) {
      client.editPerm &= ~permissionMapping._clipboard;
      if (client.editClipboard) client.editPerm |= permissionMapping._clipboard;
    },
    hasClipboard(perm) {
      return (perm & permissionMapping._clipboard) !== 0;
    },
    permToStr(perm) {
      return [24, 16, 8]
        .map(shift => ((perm >> shift) & 0xFF).toString(16).toUpperCase().padStart(2, '0'))
        .join(' ');
    },
    checkPermission(perm, permission) {
      return (perm & permissionMapping[permission]) !== 0;
    },
    isSuppressed(perm, permission, suppressedBy) {
      return suppressedBy.some(suppressed => this.checkPermission(perm, suppressed));
    },
    togglePermission(client, permission) {
      client.editPerm ^= permissionMapping[permission];
      client.editClipboard = this.hasClipboard(client.editPerm);
      client.editAccessPreset = this.accessPresetFor(client.editPerm);
    },
    disconnectClient(uuid) {
      postJson('./api/clients/disconnect', {uuid})
        .catch(error => console.error('Could not disconnect the device', error))
        .finally(() => setTimeout(() => this.refreshClients(), 1000));
    },
    async unpairAll() {
      if (!window.confirm(this.i18n.t('pin.remove_all_confirm'))) return;
      this.unpairAllPressed = true;
      try {
        const body = await postJson('./api/clients/unpair-all');
        this.unpairAllStatus = body.status;
        await this.refreshClients();
      } catch (_) {
        this.unpairAllStatus = false;
      } finally {
        this.unpairAllPressed = false;
        setTimeout(() => { this.unpairAllStatus = null; }, 5000);
      }
    },
    async unpairSingle(client) {
      const name = this.displayClientName(client);
      if (!window.confirm(this.i18n.t('pin.remove_confirm', { name }))) return;

      try {
        const body = await postJson('./api/clients/unpair', {uuid: client.uuid});
        if (body.status === false) throw new Error(body.error || body.message || this.i18n.t('pin.action_failed'));
        this.showApplyMessage = true;
        await this.refreshClients();
      } catch (error) {
        this.clientsError = this.i18n.t('pin.remove_failed', { name, error: error.message || error });
      }
    },
    async refreshClients(detectPairing = false) {
      try {
        const body = await getJson('./api/clients/list');

        // The pairing poll calls this every two seconds. Keep the entry the user is editing
        // or saving, so a refresh does not discard an open device editor.
        const inProgress = new Map(
          this.clients.filter(client => client.editing || client.saving).map(client => [client.uuid, client])
        );
        const clients = (body.named_certs || []).map(({
          name,
          uuid,
          perm,
          connected
        }) => {
          const kept = inProgress.get(uuid);
          if (kept) {
            kept.connected = connected;
            return kept;
          }
          return {
            name,
            uuid,
            perm: Number.parseInt(perm, 10),
            connected,
            editing: false,
            saving: false,
            saveError: ''
          };
        });

        this.clients = clients;
        this.clientsLoaded = true;
        this.clientsError = '';
        if (!clients.includes(currentEditingClient)) currentEditingClient = null;

        if (detectPairing && this.pairingActive) {
          const initialIds = new Set(this.pairingInitialClientIds);
          const newlyPaired = clients.find(client => !initialIds.has(client.uuid));
          if (newlyPaired) await this.completeArtemisPairing(newlyPaired);
        }
        return clients;
      } catch (error) {
        this.clientsLoaded = true;
        this.clientsError = this.i18n.t('pin.load_failed', { error: error.message || error });
        return this.clients;
      }
    }
  }
}
</script>
