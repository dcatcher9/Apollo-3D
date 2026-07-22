<script setup>
import { computed, getCurrentInstance, onBeforeUnmount, onMounted, ref } from 'vue'

defineProps({
  compact: {
    type: Boolean,
    default: false,
  },
})

const themes = [
  { value: 'light', label: 'Light', icon: 'fa-sun' },
  { value: 'dark', label: 'Dark', icon: 'fa-moon' },
  { value: 'auto', label: 'System', icon: 'fa-circle-half-stroke' },
]

const instanceId = `apollo-theme-${getCurrentInstance().uid}`
const selectedTheme = ref(localStorage.getItem('theme') || 'auto')
const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)')

const activeTheme = computed(() => themes.find(theme => theme.value === selectedTheme.value) || themes[2])

const applyTheme = theme => {
  const resolvedTheme = theme === 'auto' ? (mediaQuery.matches ? 'dark' : 'light') : theme
  document.documentElement.setAttribute('data-bs-theme', resolvedTheme)
}

const chooseTheme = theme => {
  selectedTheme.value = theme
  localStorage.setItem('theme', theme)
  applyTheme(theme)
  window.dispatchEvent(new CustomEvent('apollo-theme-change', { detail: theme }))
}

const handleSystemThemeChange = () => {
  if (selectedTheme.value === 'auto') applyTheme('auto')
}

const handleThemeEvent = event => {
  if (!event.detail || event.detail === selectedTheme.value) return
  selectedTheme.value = event.detail
  applyTheme(event.detail)
}

onMounted(() => {
  applyTheme(selectedTheme.value)
  mediaQuery.addEventListener('change', handleSystemThemeChange)
  window.addEventListener('apollo-theme-change', handleThemeEvent)
})

onBeforeUnmount(() => {
  mediaQuery.removeEventListener('change', handleSystemThemeChange)
  window.removeEventListener('apollo-theme-change', handleThemeEvent)
})
</script>

<template>
  <div class="dropdown apollo-theme-toggle" :class="{ 'is-compact': compact }">
    <button
      class="apollo-theme-button"
      type="button"
      data-bs-toggle="dropdown"
      aria-expanded="false"
      :aria-controls="instanceId"
      :aria-label="`Appearance: ${activeTheme.label}`"
      title="Change appearance">
      <i class="fa-solid fa-fw" :class="activeTheme.icon" aria-hidden="true"></i>
      <span v-if="!compact">Appearance</span>
      <i v-if="!compact" class="fa-solid fa-chevron-down apollo-theme-chevron" aria-hidden="true"></i>
    </button>
    <ul :id="instanceId" class="dropdown-menu apollo-theme-menu">
      <li v-for="theme in themes" :key="theme.value">
        <button
          type="button"
          class="dropdown-item apollo-theme-option"
          :class="{ active: selectedTheme === theme.value }"
          :aria-pressed="selectedTheme === theme.value"
          @click="chooseTheme(theme.value)">
          <i class="fa-solid fa-fw" :class="theme.icon" aria-hidden="true"></i>
          <span>{{ theme.label }}</span>
          <i v-if="selectedTheme === theme.value" class="fa-solid fa-check ms-auto" aria-hidden="true"></i>
        </button>
      </li>
    </ul>
  </div>
</template>
