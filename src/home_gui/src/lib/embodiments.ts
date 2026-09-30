import type { Language } from './types';

export function capabilityLabel(name: string): string {
  const leaf = name.split('.').at(-1) ?? name;
  const words = leaf.replace(/[_-]+/g, ' ');
  return words.charAt(0).toLocaleUpperCase() + words.slice(1);
}

export function spaceLabel(id: string | null): string {
  if (!id) return '';
  const leaf = id.split(':').at(-1) ?? id;
  const words = leaf.replace(/[_-]+/g, ' ');
  return words.charAt(0).toLocaleUpperCase() + words.slice(1);
}

export function relativeTime(value: string | null, language: Language, now = Date.now()): string {
  if (!value) return '';
  const time = Date.parse(value);
  if (!Number.isFinite(time)) return value;
  const seconds = Math.round((time - now) / 1000);
  const absolute = Math.abs(seconds);
  const unit: Intl.RelativeTimeFormatUnit = absolute < 60 ? 'second'
    : absolute < 3600 ? 'minute'
      : absolute < 86400 ? 'hour' : 'day';
  const divisor = unit === 'second' ? 1 : unit === 'minute' ? 60 : unit === 'hour' ? 3600 : 86400;
  return new Intl.RelativeTimeFormat(language, { numeric: 'auto' }).format(
    Math.round(seconds / divisor), unit,
  );
}

export function dateTime(value: string | null, language: Language): string {
  if (!value) return '—';
  const time = Date.parse(value);
  return Number.isFinite(time)
    ? new Intl.DateTimeFormat(language, { dateStyle: 'medium', timeStyle: 'medium' }).format(time)
    : value;
}

export function physicalValue(value: { value: number; p95: number }, unit: string): string {
  return `${value.value.toFixed(3)} ± ${value.p95.toFixed(3)} ${unit}`;
}
