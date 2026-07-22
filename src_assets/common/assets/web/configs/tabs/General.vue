<script setup>
import { ref, onMounted } from 'vue'
import Checkbox from '../../Checkbox.vue'

const props = defineProps({
  config: Object,
  globalPrepCmd: Array
})

const config = ref(props.config)
const globalPrepCmd = ref(props.globalPrepCmd)

const prepCmdTemplate = {
  do: "",
  undo: "",
}

function addCmd(cmdArr, template, idx) {
  const _tpl = Object.assign({}, template);

  _tpl.elevated = false;
  if (idx < 0) {
    cmdArr.push(_tpl);
  } else {
    cmdArr.splice(idx, 0, _tpl);
  }
}

function removeCmd(cmdArr, index) {
  cmdArr.splice(index,1)
}

onMounted(() => {
  // Set default value for enable_pairing if not present
  if (config.value.enable_pairing === undefined) {
    config.value.enable_pairing = "enabled"
  }
})
</script>

<template>
  <div id="general" class="config-page">
    <!-- Locale -->
    <div class="mb-3">
      <label for="locale" class="form-label">{{ $t('config.locale') }}</label>
      <select id="locale" class="form-select" v-model="config.locale">
        <option value="bg">Български (Bulgarian)</option>
        <option value="cs">Čeština (Czech)</option>
        <option value="de">Deutsch (German)</option>
        <option value="en">English</option>
        <option value="en_GB">English, UK</option>
        <option value="en_US">English, US</option>
        <option value="es">Español (Spanish)</option>
        <option value="fr">Français (French)</option>
        <option value="hu">Magyar (Hungarian)</option>
        <option value="it">Italiano (Italian)</option>
        <option value="ja">日本語 (Japanese)</option>
        <option value="ko">한국어 (Korean)</option>
        <option value="pl">Polski (Polish)</option>
        <option value="pt">Português (Portuguese)</option>
        <option value="pt_BR">Português, Brasileiro (Portuguese, Brazilian)</option>
        <option value="ru">Русский (Russian)</option>
        <option value="sv">svenska (Swedish)</option>
        <option value="tr">Türkçe (Turkish)</option>
        <option value="uk">Українська (Ukranian)</option>
        <option value="vi">Tiếng Việt (Vietnamese)</option>
        <option value="zh">简体中文 (Chinese Simplified)</option>
        <option value="zh_TW">繁體中文 (Chinese Traditional)</option>
      </select>
      <div class="form-text">{{ $t('config.locale_desc') }}</div>
    </div>

    <!-- Apollo XR Name -->
    <div class="mb-3">
      <label for="sunshine_name" class="form-label">{{ $t('config.sunshine_name') }}</label>
      <input type="text" class="form-control" id="sunshine_name" placeholder="Apollo XR"
             v-model="config.sunshine_name" />
      <div class="form-text">{{ $t('config.sunshine_name_desc') }}</div>
    </div>

    <!-- Log Level -->
    <div class="mb-3">
      <label for="min_log_level" class="form-label">{{ $t('config.min_log_level') }}</label>
      <select id="min_log_level" class="form-select" v-model="config.min_log_level">
        <option value="0">{{ $t('config.min_log_level_0') }}</option>
        <option value="1">{{ $t('config.min_log_level_1') }}</option>
        <option value="2">{{ $t('config.min_log_level_2') }}</option>
        <option value="3">{{ $t('config.min_log_level_3') }}</option>
        <option value="4">{{ $t('config.min_log_level_4') }}</option>
        <option value="5">{{ $t('config.min_log_level_5') }}</option>
        <option value="6">{{ $t('config.min_log_level_6') }}</option>
      </select>
      <div class="form-text">{{ $t('config.min_log_level_desc') }}</div>
    </div>

    <!-- Runtime diagnostics -->
    <Checkbox class="mb-3"
              id="diagnostics"
              locale-prefix="config"
              v-model="config.diagnostics"
              default="false"
    ></Checkbox>

    <!-- Global Preparation Commands -->
    <div id="global_prep_cmd" class="mb-3 d-flex flex-column">
      <label class="form-label">{{ $t('config.global_prep_cmd') }}</label>
      <div class="form-text pre-wrap">{{ $t('config.global_prep_cmd_desc') }}</div>
      <table class="table" v-if="globalPrepCmd.length > 0">
        <thead>
        <tr>
          <th scope="col"><i class="fas fa-play"></i> {{ $t('_common.do_cmd') }}</th>
          <th scope="col"><i class="fas fa-undo"></i> {{ $t('_common.undo_cmd') }}</th>
          <th scope="col">
            <i class="fas fa-shield-alt"></i> {{ $t('_common.run_as') }}
          </th>
          <th scope="col"></th>
        </tr>
        </thead>
        <tbody>
        <tr v-for="(c, i) in globalPrepCmd" :key="`prep-cmd-${i}`">
          <td>
            <input type="text" class="form-control monospace" v-model="c.do" />
          </td>
          <td>
            <input type="text" class="form-control monospace" v-model="c.undo" />
          </td>
          <td class="align-middle">
            <Checkbox :id="'prep-cmd-admin-' + i"
                      label="_common.elevated"
                      desc=""
                      default="false"
                      v-model="c.elevated"
            ></Checkbox>
          </td>
          <td class="text-end">
            <button class="btn btn-danger me-2" @click="removeCmd(globalPrepCmd, i)">
              <i class="fas fa-trash"></i>
            </button>
            <button class="btn btn-success" @click="addCmd(globalPrepCmd, prepCmdTemplate, i)">
              <i class="fas fa-plus"></i>
            </button>
          </td>
        </tr>
        </tbody>
      </table>
      <button class="ms-0 mt-2 btn btn-success" style="margin: 0 auto" @click="addCmd(globalPrepCmd, prepCmdTemplate, -1)">
        &plus; {{ $t('config.add') }}
      </button>
    </div>

    <!-- Enable Pairing -->
    <Checkbox class="mb-3"
              id="enable_pairing"
              locale-prefix="config"
              v-model="config.enable_pairing"
              default="true"
    ></Checkbox>

    <!-- Enable Discovery -->
    <Checkbox class="mb-3"
              id="enable_discovery"
              locale-prefix="config"
              v-model="config.enable_discovery"
              default="true"
    ></Checkbox>

    <!-- Enable system tray -->
    <Checkbox class="mb-3"
              id="system_tray"
              locale-prefix="config"
              v-model="config.system_tray"
              default="true"
    ></Checkbox>

    <!-- Hide Tray Controls -->
    <Checkbox class="mb-3"
              id="hide_tray_controls"
              locale-prefix="config"
              v-model="config.hide_tray_controls"
              default="false"
    ></Checkbox>
  </div>
</template>

<style scoped>

</style>
