/**
 * @file tests/unit/test_crypto.cpp
 * @brief Test src/crypto.*.
 */
#include "../tests_common.h"

#include <openssl/x509.h>
#include <src/crypto.h>

namespace {
  crypto::aes_t fixed_aes_value(std::uint8_t first) {
    crypto::aes_t value(16, 0);
    value.front() = first;
    return value;
  }
}  // namespace

TEST(CryptoTest, GeneratedCredentialsExposeSubjectAndVerifySignatures) {
  constexpr std::string_view common_name = "Apollo Test Host";
  constexpr std::string_view payload = "payload";

  auto creds = crypto::gen_creds(common_name, 2048);
  ASSERT_FALSE(creds.x509.empty());
  ASSERT_FALSE(creds.pkey.empty());

  auto cert = crypto::x509(creds.x509);
  auto pkey = crypto::pkey(creds.pkey);
  ASSERT_NE(cert.get(), nullptr);
  ASSERT_NE(pkey.get(), nullptr);

  const auto subject = X509_get_subject_name(cert.get());
  ASSERT_NE(subject, nullptr);

  const auto common_name_index = X509_NAME_get_index_by_NID(subject, NID_commonName, -1);
  ASSERT_GE(common_name_index, 0);

  const auto common_name_entry = X509_NAME_get_entry(subject, common_name_index);
  ASSERT_NE(common_name_entry, nullptr);

  const auto common_name_data = X509_NAME_ENTRY_get_data(common_name_entry);
  ASSERT_NE(common_name_data, nullptr);

  const std::string_view parsed_common_name {
    reinterpret_cast<const char *>(ASN1_STRING_get0_data(common_name_data)),
    static_cast<std::size_t>(ASN1_STRING_length(common_name_data))
  };
  ASSERT_EQ(parsed_common_name, common_name);

  const auto issuer = X509_get_issuer_name(cert.get());
  ASSERT_NE(issuer, nullptr);
  ASSERT_EQ(X509_NAME_cmp(subject, issuer), 0);
  ASSERT_EQ(X509_verify(cert.get(), X509_get0_pubkey(cert.get())), 1);

  ASSERT_FALSE(crypto::signature(cert).empty());

  const auto signed_payload = crypto::sign256(pkey, payload);
  ASSERT_FALSE(signed_payload.empty());
  ASSERT_TRUE(crypto::verify256(cert, payload, {reinterpret_cast<const char *>(signed_payload.data()), signed_payload.size()}));
}

TEST(CryptoTest, GcmDecryptReinitializationFailureReturnsError) {
  const auto key = fixed_aes_value(0x11);
  auto iv = fixed_aes_value(0x22);
  constexpr std::string_view plaintext = "authenticated input";

  crypto::cipher::gcm_t cipher {key, false};
  std::vector<std::uint8_t> tagged_cipher(
    crypto::cipher::tag_size + plaintext.size()
  );
  ASSERT_EQ(
    cipher.encrypt(plaintext, tagged_cipher.data(), &iv),
    static_cast<int>(plaintext.size())
  );

  std::vector<std::uint8_t> decrypted;
  const std::string_view tagged_view {
    reinterpret_cast<const char *>(tagged_cipher.data()),
    tagged_cipher.size()
  };
  ASSERT_EQ(cipher.decrypt(tagged_view, decrypted, &iv), 0);
  ASSERT_EQ(std::string_view(reinterpret_cast<const char *>(decrypted.data()), decrypted.size()), plaintext);

  // Keep a non-null context while removing its cipher. This deterministically forces the exact
  // EVP_DecryptInit_ex(nullptr cipher) reinitialization branch to fail.
  ASSERT_EQ(EVP_CIPHER_CTX_reset(cipher.decrypt_ctx.get()), 1);
  EXPECT_EQ(cipher.decrypt(tagged_view, decrypted, &iv), -1);
}

TEST(CryptoTest, GcmDecryptCanSplitAuthenticatedPrefixWithoutPayloadMemmove) {
  const auto key = fixed_aes_value(0x51);
  auto iv = fixed_aes_value(0x62);
  constexpr std::array<std::uint8_t, 9> plaintext {
    0x07, 0x03, 0x05, 0x00, 9, 8, 7, 6, 5
  };

  crypto::cipher::gcm_t cipher {key, false};
  std::vector<std::uint8_t> tagged_cipher(
    crypto::cipher::tag_size + plaintext.size()
  );
  ASSERT_EQ(
    cipher.encrypt(
      {reinterpret_cast<const char *>(plaintext.data()), plaintext.size()},
      tagged_cipher.data(),
      &iv
    ),
    static_cast<int>(plaintext.size())
  );

  std::array<std::uint8_t, 4> header {};
  std::vector<std::uint8_t> payload;
  ASSERT_EQ(
    cipher.decrypt(
      {reinterpret_cast<const char *>(tagged_cipher.data()), tagged_cipher.size()},
      header,
      payload,
      &iv
    ),
    0
  );
  EXPECT_TRUE(std::equal(header.begin(), header.end(), plaintext.begin()));
  EXPECT_TRUE(std::equal(payload.begin(), payload.end(), plaintext.begin() + header.size()));

  // The split overload must be byte-equivalent to the existing contiguous output.
  std::vector<std::uint8_t> contiguous;
  ASSERT_EQ(
    cipher.decrypt(
      {reinterpret_cast<const char *>(tagged_cipher.data()), tagged_cipher.size()},
      contiguous,
      &iv
    ),
    0
  );
  EXPECT_TRUE(std::equal(contiguous.begin(), contiguous.end(), plaintext.begin()));
}

TEST(CryptoTest, SplitGcmDecryptClearsUnauthenticatedOutputOnTagFailure) {
  const auto key = fixed_aes_value(0x71);
  auto iv = fixed_aes_value(0x82);
  constexpr std::string_view plaintext = "headpayload";

  crypto::cipher::gcm_t cipher {key, false};
  std::vector<std::uint8_t> tagged_cipher(
    crypto::cipher::tag_size + plaintext.size()
  );
  ASSERT_EQ(
    cipher.encrypt(plaintext, tagged_cipher.data(), &iv),
    static_cast<int>(plaintext.size())
  );
  tagged_cipher.front() ^= 0x80;

  std::array<std::uint8_t, 4> header {1, 1, 1, 1};
  std::vector<std::uint8_t> payload {2, 2, 2};
  EXPECT_EQ(
    cipher.decrypt(
      {reinterpret_cast<const char *>(tagged_cipher.data()), tagged_cipher.size()},
      header,
      payload,
      &iv
    ),
    -1
  );
  EXPECT_TRUE(std::ranges::all_of(header, [](std::uint8_t byte) { return byte == 0; }));
  EXPECT_TRUE(payload.empty());
}

TEST(CryptoTest, CbcEncryptReinitializationFailureReturnsError) {
  const auto key = fixed_aes_value(0x33);
  auto iv = fixed_aes_value(0x44);
  constexpr std::string_view plaintext = "sixteen-byte-msg";

  crypto::cipher::cbc_t cipher {key, false};
  std::vector<std::uint8_t> encrypted(plaintext.size() + EVP_MAX_BLOCK_LENGTH);
  ASSERT_EQ(
    cipher.encrypt(plaintext, encrypted.data(), &iv),
    static_cast<int>(plaintext.size())
  );

  // As above, preserve the allocated context but clear its configured cipher.
  ASSERT_EQ(EVP_CIPHER_CTX_reset(cipher.encrypt_ctx.get()), 1);
  EXPECT_EQ(cipher.encrypt(plaintext, encrypted.data(), &iv), -1);
}
