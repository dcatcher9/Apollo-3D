/**
 * Browser boundary for the native offline-SBS job manager inside Sunshine 3D.
 *
 * Keep endpoint details here so the page stays a presentation layer; HTTP and its failures are
 * handled by the shared API helper. Version 1 deliberately exposes no "resume" call: an
 * interrupted job can only be restarted as a new job from its immutable settings after the job
 * manager reports the interruption and retained artifacts.
 */
import { ApiError, requestJson } from './api.js'

export const OFFLINE_SBS_JOBS_URL = '/api/offline-sbs/jobs'
export const OFFLINE_SBS_BROWSE_URL = '/api/offline-sbs/browse'

function requireJobId(jobId) {
  const normalized = String(jobId ?? '').trim()
  if (!normalized) {
    throw new ApiError('A conversion job id is required.')
  }
  return encodeURIComponent(normalized)
}

export function normalizeJobs(payload) {
  if (Array.isArray(payload)) return payload
  if (Array.isArray(payload?.jobs)) return payload.jobs
  if (Array.isArray(payload?.job)) return payload.job
  if (payload?.job && typeof payload.job === 'object') return [payload.job]
  return []
}

export async function listOfflineSbsJobs({ signal } = {}) {
  return normalizeJobs(await requestJson(OFFLINE_SBS_JOBS_URL, { signal }))
}

export async function getOfflineSbsOverview({ signal } = {}) {
  let payload
  try {
    payload = await requestJson(OFFLINE_SBS_JOBS_URL, { signal })
  } catch (error) {
    const capabilities = error?.payload?.capabilities
    if (
      error instanceof ApiError &&
      error.status === 503 &&
      capabilities?.available === false &&
      capabilities.initializing === true
    ) {
      return { jobs: [], capabilities }
    }
    throw error
  }
  return {
    jobs: normalizeJobs(payload),
    capabilities:
      payload?.capabilities && typeof payload.capabilities === 'object' ?
        payload.capabilities :
        {},
  }
}

export async function getOfflineSbsJob(jobId, { signal } = {}) {
  const payload = await requestJson(
    `${OFFLINE_SBS_JOBS_URL}/${requireJobId(jobId)}`,
    { signal },
  )
  return payload?.job && typeof payload.job === 'object' ? payload.job : null
}

export async function browseOfflineSbsFiles(
  { path = '', type = 'file', signal } = {},
) {
  if (!['any', 'directory', 'file'].includes(type)) {
    throw new ApiError('The file browser type is invalid.')
  }
  return requestJson(OFFLINE_SBS_BROWSE_URL, {
    method: 'POST',
    body: { path, type },
    signal,
  })
}

export async function createOfflineSbsJob(settings, { signal } = {}) {
  return requestJson(OFFLINE_SBS_JOBS_URL, {
    method: 'POST',
    body: settings,
    signal,
  })
}

export async function cancelOfflineSbsJob(jobId, { signal } = {}) {
  return requestJson(`${OFFLINE_SBS_JOBS_URL}/${requireJobId(jobId)}/cancel`, {
    method: 'POST',
    signal,
  })
}

export async function clearOfflineSbsJob(jobId, { signal } = {}) {
  return requestJson(`${OFFLINE_SBS_JOBS_URL}/${requireJobId(jobId)}`, {
    method: 'DELETE',
    signal,
  })
}
