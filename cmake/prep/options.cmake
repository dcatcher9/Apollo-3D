# Publisher Metadata
set(SUNSHINE_PUBLISHER_NAME "SudoMaker"
        CACHE STRING "The name of the publisher (or fork developer) of the application.")
set(SUNSHINE_PUBLISHER_WEBSITE "https://www.sudomaker.com"
        CACHE STRING "The URL of the publisher's website.")
set(SUNSHINE_PUBLISHER_ISSUE_URL "https://github.com/ClassicOldSong/Apollo/issues"
        CACHE STRING "The URL of the publisher's support site or issue tracker.
        If you provide a modified version of Sunshine, we kindly request that you use your own url.")

option(BUILD_DOCS "Build documentation" OFF)
option(BUILD_TESTS "Build tests" OFF)
option(NPM_OFFLINE "Use offline npm packages. You must ensure packages are in your npm cache." OFF)

option(BUILD_WERROR "Enable -Werror flag." OFF)

option(SUNSHINE_ENABLE_TRAY "Enable system tray icon." ON)

option(BOOST_USE_STATIC "Use static Boost libraries." ON)
