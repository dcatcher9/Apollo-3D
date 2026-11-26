import {createI18n} from "vue-i18n";
import {getJson} from './api'

// Import only the fallback language files
import en from './public/assets/locale/en.json'

export default async function() {
    let locale = "en";
    try {
        locale = (await getJson("./api/configLocale")).locale ?? "en";
    } catch (e) {
        // Every page waits for this before it renders, so a failed read falls back to English.
        console.error("Failed to read the locale setting", e);
    }
    document.querySelector('html').setAttribute('lang', locale);
    let messages = {
        en
    };
    try {
        if (locale !== 'en') {
            let r = await (await fetch(`./assets/locale/${locale}.json`, { credentials: 'include' })).json();
            messages[locale] = r;
        }
    } catch (e) {
        console.error("Failed to download translations", e);
    }
    const i18n = createI18n({
        locale: locale, // set locale
        fallbackLocale: 'en', // set fallback locale
        messages: messages
    })
    return i18n;
}
