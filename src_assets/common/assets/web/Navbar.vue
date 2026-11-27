<script setup>
import { computed, onMounted, ref } from 'vue'
import ThemeToggle from './ThemeToggle.vue'
import { getJson } from './api'

const version = ref('')

const primaryItems = [
  { id: 'overview', labelKey: 'navbar.overview', href: './', icon: 'fa-house' },
  { id: 'devices', labelKey: 'navbar.devices', href: './pin', icon: 'fa-vr-cardboard' },
  { id: 'library', labelKey: 'navbar.library', href: './apps', icon: 'fa-table-cells-large' },
  { id: 'conversion', labelKey: 'navbar.convert', href: './offline-conversion', icon: 'fa-film' },
  { id: 'settings', labelKey: 'navbar.settings', href: './config', icon: 'fa-sliders' },
]

const currentPage = computed(() => {
  const path = window.location.pathname.replace(/\/$/, '')
  const page = path.substring(path.lastIndexOf('/') + 1).replace(/\.html$/, '')
  if (!page || page === 'index') return 'overview'
  if (page === 'pin') return 'devices'
  if (page === 'apps') return 'library'
  if (page === 'offline-conversion') return 'conversion'
  if (page === 'config') return 'settings'
  if (page === 'troubleshooting') return 'help'
  return ''
})

onMounted(async () => {
  try {
    const config = await getJson('./api/config')
    version.value = config.version || ''
  } catch (error) {
    console.debug('Sunshine 3D version is unavailable', error)
  }
})
</script>

<template>
  <aside class="apollo-sidebar" :aria-label="$t('navbar.sidebar_label')">
    <a class="apollo-brand" href="./" :aria-label="$t('navbar.brand_label')">
      <span class="apollo-brand-mark">
        <img src="/images/logo-sunshine3d-45.png" alt="">
      </span>
      <span class="apollo-brand-copy">
        <strong>Sunshine 3D</strong>
        <small>{{ $t('navbar.host') }}</small>
      </span>
    </a>

    <nav class="apollo-primary-nav" :aria-label="$t('navbar.main_navigation')">
      <a
        v-for="item in primaryItems"
        :key="item.id"
        class="apollo-nav-link"
        :class="{ active: currentPage === item.id }"
        :href="item.href"
        :aria-current="currentPage === item.id ? 'page' : undefined">
        <i class="fa-solid fa-fw" :class="item.icon" aria-hidden="true"></i>
        <span>{{ $t(item.labelKey) }}</span>
      </a>
    </nav>

    <div class="apollo-sidebar-footer">
      <a
        class="apollo-nav-link apollo-help-link"
        :class="{ active: currentPage === 'help' }"
        href="./troubleshooting"
        :aria-current="currentPage === 'help' ? 'page' : undefined">
        <i class="fa-solid fa-fw fa-circle-question" aria-hidden="true"></i>
        <span>{{ $t('navbar.help_logs') }}</span>
      </a>
      <ThemeToggle />
      <p v-if="version" class="apollo-version">Sunshine 3D {{ version }}</p>
    </div>
  </aside>

  <header class="apollo-mobile-header">
    <a class="apollo-mobile-brand" href="./" :aria-label="$t('navbar.brand_label')">
      <img src="/images/logo-sunshine3d-45.png" alt="">
      <strong>Sunshine 3D</strong>
    </a>
    <div class="apollo-mobile-actions">
      <a class="apollo-mobile-utility" href="./troubleshooting#logs" :aria-label="$t('navbar.open_debug_log')" :title="$t('navbar.debug_log')">
        <i class="fa-solid fa-file-lines" aria-hidden="true"></i>
      </a>
      <ThemeToggle compact />
    </div>
  </header>

  <nav class="apollo-bottom-nav" :aria-label="$t('navbar.main_navigation')">
    <a
      v-for="item in primaryItems"
      :key="item.id"
      class="apollo-bottom-link"
      :class="{ active: currentPage === item.id }"
      :href="item.href"
      :aria-current="currentPage === item.id ? 'page' : undefined">
      <i class="fa-solid" :class="item.icon" aria-hidden="true"></i>
      <span>{{ $t(item.labelKey) }}</span>
    </a>
  </nav>
</template>
