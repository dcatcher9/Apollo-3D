<template>
    <section class="apollo-resource-card">
        <div>
            <p class="apollo-eyebrow">Support</p>
            <h2>{{ $t('resource_card.resources') }}</h2>
            <p>{{ $t('resource_card.resources_desc') }}</p>
            <div class="apollo-resource-actions">
                <!-- <a class="btn btn-success m-1" href="https://app.lizardbyte.dev" target="_blank">
                    {{ $t('resource_card.lizardbyte_website') }}</a> -->
                <a class="btn btn-apollo-quiet" href="https://github.com/ClassicOldSong/Apollo/discussions" target="_blank">
                    <i class="fab fa-fw fa-github"></i> {{ $t('resource_card.github_discussions') }}</a>
                <a class="btn btn-apollo-quiet" href="https://github.com/ClassicOldSong/Apollo/wiki" target="_blank">
                    <i class="fab fa-fw fa-github"></i> {{ $t('resource_card.github_wiki') }}</a>
                <a class="btn btn-apollo-quiet" href="https://github.com/ClassicOldSong/Apollo/wiki/Stuttering-Clinic" target="_blank">
                    <i class="fab fa-fw fa-github"></i> {{ $t('resource_card.github_stuttering_clinic') }}</a>
            </div>
        </div>
    </section>
    <!-- Legal -->
    <section class="apollo-resource-card">
        <div>
            <p class="apollo-eyebrow">Project</p>
            <h2>{{ $t('resource_card.legal') }}</h2>
            <p>{{ $t('resource_card.legal_desc') }}</p>
            <div class="apollo-resource-actions">
                <a class="btn btn-apollo-quiet" href="https://github.com/ClassicOldSong/Apollo/blob/master/LICENSE"
                    target="_blank">
                    <i class="fas fa-fw fa-file-alt"></i> {{ $t('resource_card.license') }}</a>
                <a class="btn btn-apollo-quiet" href="https://github.com/ClassicOldSong/Apollo/blob/master/NOTICE"
                    target="_blank">
                    <i class="fas fa-fw fa-exclamation"></i> {{ $t('resource_card.third_party_notice') }}</a>
            </div>
        </div>
    </section>
</template>
