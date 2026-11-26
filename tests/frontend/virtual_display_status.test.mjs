import assert from 'node:assert/strict'
import test from 'node:test'
import { virtualDisplayStatus } from '../../src_assets/common/assets/web/virtual-display-status.js'

test('every driver state the host reports has its own label', () => {
  const labels = [0, 1, -1, -2, -3].map(code => virtualDisplayStatus(code).label)
  assert.deepEqual(labels, ['Ready', 'Checking', 'Not initialized', 'Update required', 'Needs attention'])
  assert.equal(new Set(labels).size, labels.length)
})

test('string and numeric codes read the same state', () => {
  for (const code of [0, 1, -1, -2, -3]) {
    assert.deepEqual(virtualDisplayStatus(String(code)), virtualDisplayStatus(code))
  }
})

test('only the ready state is a success', () => {
  assert.equal(virtualDisplayStatus(0).tone, 'success')
  for (const code of [1, -1, -2, -3]) {
    assert.notEqual(virtualDisplayStatus(code).tone, 'success')
  }
})

test('an unreported or unknown value reads as checking', () => {
  for (const code of [undefined, null, '', Number.NaN, 7, 'unknown']) {
    assert.equal(virtualDisplayStatus(code).label, 'Checking')
  }
})
