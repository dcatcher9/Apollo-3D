#include "src/config.h"

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

TEST(ConfigParsing, PreservesFullMonitorIdentityThroughFileAndSurfaceParsing) {
  const std::string path = R"(\\?\DISPLAY#GSMC0C8#5&12ddb9ee&1&UID4352#{e6f07b5f-ee97-4a90-b076-33f57bf4eaa7})";
  auto parsed = config::parse_config(
    "local_monitor_enabled = enabled\r\n"
    "local_monitor_display_id = " + path + "  # selected physical monitor\r\n"
    "local_monitor_emitter_port = COM7\r\n"
  );
  ASSERT_TRUE(parsed.contains("local_monitor_display_id"));
  EXPECT_EQ(parsed.at("local_monitor_display_id"), path);
  const auto monitor = config::parse_local_monitor_config(parsed);
  EXPECT_TRUE(monitor.enabled);
  EXPECT_EQ(monitor.display_id, path);
}

TEST(ConfigParsing, PreservesLiteralUrlFragmentsButRemovesWhitespaceDelimitedComments) {
  const auto parsed = config::parse_config(
    "# comment-only line\n"
    "source_url = https://example.test/watch#chapter#anchor\n"
    "sunshine_name = XR host  # user-facing name\n"
    "output_name = monitor#instance\t# selected connector\n"
  );
  ASSERT_EQ(parsed.size(), 3u);
  EXPECT_EQ(parsed.at("source_url"), "https://example.test/watch#chapter#anchor");
  EXPECT_EQ(parsed.at("sunshine_name"), "XR host");
  EXPECT_EQ(parsed.at("output_name"), "monitor#instance");
}

TEST(ConfigParsing, PreservesQuotedHashesAndEscapedQuotesInMultilineJsonLists) {
  const auto parsed = config::parse_config(R"CONFIG(
global_prep_cmd = [
  # a complete comment line between JSON array elements
  {"do":"echo https://example.test/video#chapter", "undo":"echo value # literal"}, # actual inline comment
  {"do":"echo \"quoted # fragment\"", "undo":"C:\\Games\\name#edition"}
]
sunshine_name = Still parsed # trailing comment
)CONFIG");
  ASSERT_TRUE(parsed.contains("global_prep_cmd"));
  const auto commands = nlohmann::json::parse(parsed.at("global_prep_cmd"));
  ASSERT_EQ(commands.size(), 2u);
  EXPECT_EQ(commands[0].at("do"), "echo https://example.test/video#chapter");
  EXPECT_EQ(commands[0].at("undo"), "echo value # literal");
  EXPECT_EQ(commands[1].at("do"), "echo \"quoted # fragment\"");
  EXPECT_EQ(commands[1].at("undo"), R"(C:\Games\name#edition)");
  EXPECT_EQ(parsed.at("sunshine_name"), "Still parsed");
}

TEST(ConfigParsing, BracketsInsideQuotedListStringsAreText) {
  const auto parsed = config::parse_config(R"CONFIG(
global_prep_cmd = [{"do":"echo ]","undo":"echo [ \"[\" ["}]
sunshine_name = Still parsed
)CONFIG");
  ASSERT_TRUE(parsed.contains("global_prep_cmd"));
  const auto commands = nlohmann::json::parse(parsed.at("global_prep_cmd"));
  ASSERT_EQ(commands.size(), 1u);
  EXPECT_EQ(commands[0].at("do"), "echo ]");
  EXPECT_EQ(commands[0].at("undo"), "echo [ \"[\" [");
  EXPECT_EQ(parsed.at("sunshine_name"), "Still parsed");
}

TEST(ConfigParsing, AnUnclosedListLosesOnlyItsOwnOption) {
  const auto parsed = config::parse_config(
    "resolutions = [1920x1080, 1280x720\n"
    "sunshine_name = Still parsed\n"
    "fps = [60, 90]\n"
  );
  EXPECT_FALSE(parsed.contains("resolutions"));
  EXPECT_EQ(parsed.at("sunshine_name"), "Still parsed");
  EXPECT_EQ(parsed.at("fps"), "[60, 90]");
}

TEST(ConfigParsing, ALoneCarriageReturnEndsALineWithoutEatingTheNextCharacter) {
  const auto parsed = config::parse_config("sunshine_name = one\rfps = [60]\r");
  EXPECT_EQ(parsed.at("sunshine_name"), "one");
  EXPECT_EQ(parsed.at("fps"), "[60]");
}

TEST(ConfigParsing, QuotedScalarHashSurvivesBeforeAnActualInlineComment) {
  const auto parsed = config::parse_config("sunshine_name = \"Display # one\" # comment\r\n");
  EXPECT_EQ(parsed.at("sunshine_name"), "\"Display # one\"");
}

TEST(ConfigParsing, BlankAndCommentOnlyCrLfLinesKeepOrdinaryOptionsIntact) {
  const auto parsed = config::parse_config(
    "\r\n"
    "  # indented whole-line comment\r\n"
    "\t\r\n"
    "sunshine_name = XR host\r\n"
    "# another whole-line comment\r\n"
    "max_bitrate = 0\r\n"
    "output_name = ordinary-monitor\r\n"
  );
  ASSERT_EQ(parsed.size(), 3u);
  EXPECT_EQ(parsed.at("sunshine_name"), "XR host");
  EXPECT_EQ(parsed.at("max_bitrate"), "0");
  EXPECT_EQ(parsed.at("output_name"), "ordinary-monitor");
}
