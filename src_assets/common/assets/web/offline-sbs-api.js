/**
 * Browser boundary for the native offline-SBS job manager inside Sunshine 3D.
 *
 * Keep endpoint details here so the page stays a presentation layer. Version 1 deliberately
 * exposes no "resume" call: an interrupted job can only be restarted as a new job from its
 * immutable settings after the job manager reports the interruption and retained artifacts.
 */
export const OFFLINE_SBS_JOBS_URL = '/api/offline-sbs/jobs'

export class OfflineSbsApiError extends Error {
  constructor(message, { status = 0, payload = null } = {}) {
    super(message)
    this.name = 'OfflineSbsApiError'
    this.status = status
    this.payload = payload
  }
}

async function readResponse(response) {
  const text = await response.text()
  if (!text) return {}

  try {
    return JSON.parse(text)
  } catch {
    throw new OfflineSbsApiError('The offline job manager returned invalid JSON.', {
      status: response.status,
    })
  }
}

async function request(url, { method = 'GET', body, signal } = {}) {
  const response = await fetch(url, {
    method,
    credentials: 'include',
    headers: {
      Accept: 'application/json',
      ...(body === undefined ? {} : { 'Content-Type': 'application/json' }),
    },
    body: body === undefined ? undefined : JSON.stringify(body),
    signal,
  })
  const payload = await readResponse(response)

  if (!response.ok || payload?.status === false) {
    const message =
      payload?.error ||
      payload?.message ||
      `The offline job manager returned HTTP ${response.status}.`
    throw new OfflineSbsApiError(message, {
      status: response.status,
      payload,
    })
  }
  return payload
}

function requireJobId(jobId) {
  const normalized = String(jobId ?? '').trim()
  if (!normalized) {
    throw new OfflineSbsApiError('A conversion job id is required.')
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
  return normalizeJobs(await request(OFFLINE_SBS_JOBS_URL, { signal }))
}

export async function getOfflineSbsOverview({ signal } = {}) {
  const payload = await request(OFFLINE_SBS_JOBS_URL, { signal })
  return {
    jobs: normalizeJobs(payload),
    capabilities:
      payload?.capabilities && typeof payload.capabilities === 'object' ?
        payload.capabilities :
        {},
  }
}

export async function getOfflineSbsJob(jobId, { signal } = {}) {
  const payload = await request(
    `${OFFLINE_SBS_JOBS_URL}/${requireJobId(jobId)}`,
    { signal },
  )
  return payload?.job && typeof payload.job === 'object' ? payload.job : null
}

export async function createOfflineSbsJob(settings, { signal } = {}) {
  return request(OFFLINE_SBS_JOBS_URL, {
    method: 'POST',
    body: settings,
    signal,
  })
}

export async function cancelOfflineSbsJob(jobId, { signal } = {}) {
  return request(`${OFFLINE_SBS_JOBS_URL}/${requireJobId(jobId)}/cancel`, {
    method: 'POST',
    signal,
  })
}

export async function downloadOfflineSbsSceneAudit(jobId, { signal } = {}) {
  const response = await fetch(
    `${OFFLINE_SBS_JOBS_URL}/${requireJobId(jobId)}/scene-audit`,
    {
      credentials: 'include',
      headers: { Accept: 'application/json' },
      signal,
    },
  )
  if (!response.ok) {
    const payload = await readResponse(response)
    throw new OfflineSbsApiError(
      payload?.error ||
        `The offline job manager returned HTTP ${response.status}.`,
      {
        status: response.status,
        payload,
      },
    )
  }
  return response.blob()
}
