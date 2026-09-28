/**
 * Clipboard writing with a graceful path for environments without the async
 * Clipboard API (plain http on a LAN, older browsers): fall back to a hidden
 * textarea and `execCommand('copy')`, and report failure instead of throwing
 * so callers can leave the value on screen for a manual copy.
 */

export async function copyText(text: string): Promise<boolean> {
  const clipboard = navigator.clipboard
  if (clipboard?.writeText) {
    try {
      await clipboard.writeText(text)
      return true
    } catch {
      // Permission denied or the document is not focused: try the fallback.
    }
  }
  try {
    const area = document.createElement('textarea')
    area.value = text
    area.setAttribute('readonly', '')
    area.style.position = 'fixed'
    area.style.top = '-1000px'
    area.style.opacity = '0'
    document.body.appendChild(area)
    area.select()
    const ok = document.execCommand('copy')
    document.body.removeChild(area)
    return ok
  } catch {
    return false
  }
}
