// The en locale's text for a message key, with vue-i18n's {name} placeholders filled in, so a
// test can assert the English text a page shows without a Vue app.
import { readFileSync } from 'node:fs'

const en = JSON.parse(readFileSync(new URL('../../src_assets/common/assets/web/public/assets/locale/en.json', import.meta.url), 'utf8'))

export function english(key, params = {}) {
  const message = key.split('.').reduce((messages, part) => messages?.[part], en)
  if (typeof message !== 'string') throw new Error(`en.json has no message ${key}`)
  return message.replace(/\{(\w+)\}/g, (placeholder, name) => (name in params ? String(params[name]) : placeholder))
}
