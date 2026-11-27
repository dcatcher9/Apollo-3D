/**
 * Host path text for the offline conversion page and its path picker. The paths are the host's
 * Windows paths as the job manager reports them, so either separator can appear.
 */

/** The folder that holds `path`, keeping a drive root such as `D:\`. */
export function parentHostPath(path) {
  const source = String(path || '').trim()
  const separatorIndex = Math.max(source.lastIndexOf('\\'), source.lastIndexOf('/'))
  if (separatorIndex < 0) return ''

  const parent = source.slice(0, separatorIndex)
  if (/^[A-Za-z]:$/.test(parent)) {
    return `${parent}${source[separatorIndex]}`
  }
  return parent || source[separatorIndex]
}

/** `name` inside the folder `root`, with the folder's own separator; empty when either is empty. */
export function joinHostPath(root, name) {
  const directory = String(root || '').trim()
  const leaf = String(name || '').trim()
  if (!directory || !leaf) return ''
  const separator = directory.includes('\\') && !directory.includes('/') ? '\\' : '/'
  return `${directory.replace(/[\\/]+$/, '')}${separator}${leaf}`
}
