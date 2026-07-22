<script setup>
import { computed, onMounted, ref } from 'vue'
import ThemeToggle from './ThemeToggle.vue'

const version = ref('')

const primaryItems = [
  { id: 'overview', label: 'Overview', href: './', icon: 'fa-house' },
  { id: 'devices', label: 'Devices', href: './pin', icon: 'fa-vr-cardboard' },
  { id: 'library', label: 'Library', href: './apps', icon: 'fa-table-cells-large' },
  { id: 'settings', label: 'Settings', href: './config', icon: 'fa-sliders' },
]

const currentPage = computed(() => {
  const path = window.location.pathname.replace(/\/$/, '')
  const page = path.substring(path.lastIndexOf('/') + 1).replace(/\.html$/, '')
  if (!page || page === 'index') return 'overview'
  if (page === 'pin') return 'devices'
  if (page === 'apps') return 'library'
  if (page === 'config') return 'settings'
  if (page === 'troubleshooting') return 'help'
  return ''
})

onMounted(async () => {
  try {
    const response = await fetch('./api/config', { credentials: 'include' })
    if (response.ok) {
      const config = await response.json()
      version.value = config.version || ''
    }
  } catch (error) {
    console.debug('Apollo XR version is unavailable', error)
  }
})
</script>

<template>
  <aside class="apollo-sidebar" aria-label="Apollo XR navigation">
    <a class="apollo-brand" href="./" aria-label="Apollo XR overview">
      <span class="apollo-brand-mark">
        <img src="/images/logo-apollo-45.png" alt="">
      </span>
      <span class="apollo-brand-copy">
        <strong>Apollo XR</strong>
        <small>Host</small>
      </span>
    </a>

    <nav class="apollo-primary-nav" aria-label="Main navigation">
      <a
        v-for="item in primaryItems"
        :key="item.id"
        class="apollo-nav-link"
        :class="{ active: currentPage === item.id }"
        :href="item.href"
        :aria-current="currentPage === item.id ? 'page' : undefined">
        <i class="fa-solid fa-fw" :class="item.icon" aria-hidden="true"></i>
        <span>{{ item.label }}</span>
      </a>
    </nav>

    <div class="apollo-sidebar-footer">
      <a
        class="apollo-nav-link apollo-help-link"
        :class="{ active: currentPage === 'help' }"
        href="./troubleshooting"
        :aria-current="currentPage === 'help' ? 'page' : undefined">
        <i class="fa-solid fa-fw fa-circle-question" aria-hidden="true"></i>
        <span>Help &amp; Logs</span>
      </a>
      <ThemeToggle />
      <p v-if="version" class="apollo-version">Apollo XR {{ version }}</p>
    </div>
  </aside>

  <header class="apollo-mobile-header">
    <a class="apollo-mobile-brand" href="./" aria-label="Apollo XR overview">
      <img src="/images/logo-apollo-45.png" alt="">
      <strong>Apollo XR</strong>
    </a>
    <div class="apollo-mobile-actions">
      <a class="apollo-mobile-utility" href="./troubleshooting#logs" aria-label="Open debug log" title="Debug log">
        <i class="fa-solid fa-file-lines" aria-hidden="true"></i>
      </a>
      <ThemeToggle compact />
    </div>
  </header>

  <nav class="apollo-bottom-nav" aria-label="Main navigation">
    <a
      v-for="item in primaryItems"
      :key="item.id"
      class="apollo-bottom-link"
      :class="{ active: currentPage === item.id }"
      :href="item.href"
      :aria-current="currentPage === item.id ? 'page' : undefined">
      <i class="fa-solid" :class="item.icon" aria-hidden="true"></i>
      <span>{{ item.label }}</span>
    </a>
  </nav>
</template>
