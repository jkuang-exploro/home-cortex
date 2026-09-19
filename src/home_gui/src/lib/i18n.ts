import type { Language } from './types';

const copy = {
  en: {
    product: 'Home Cortex',
    tagline: 'Household steward',
    email: 'Email',
    apiKey: 'Household API key',
    signIn: 'Sign in',
    signOut: 'Sign out',
    newChat: 'New chat',
    chats: 'Chats',
    models: 'Model',
    agents: 'Household',
    bare: 'Language model',
    placeholder: 'Ask the household…',
    send: 'Send',
    empty: 'Start a conversation.',
    delete: 'Delete',
    loginHelp: 'Use an email from CORTEX_IDENTITY_MAP. The key is sent once and stored in an HttpOnly cookie.',
    needKey: 'A Cortex API key is required.',
    mapped: 'That email is not mapped to a household person.',
    thinking: 'Thinking…',
    you: 'You',
  },
  zh: {
    product: 'Home Cortex',
    tagline: '家庭管家',
    email: '邮箱',
    apiKey: '家庭 API 密钥',
    signIn: '登录',
    signOut: '退出',
    newChat: '新对话',
    chats: '对话',
    models: '模型',
    agents: '家庭',
    bare: '语言模型',
    placeholder: '问家里的事…',
    send: '发送',
    empty: '开始一段对话。',
    delete: '删除',
    loginHelp: '使用 CORTEX_IDENTITY_MAP 中的邮箱。密钥只发送一次，之后保存在 HttpOnly cookie。',
    needKey: '需要 Cortex API 密钥。',
    mapped: '该邮箱未映射到家庭成员。',
    thinking: '正在思考…',
    you: '您',
  },
} as const;

export type Copy = (typeof copy)[Language];

export function t(language: Language): Copy {
  return copy[language] ?? copy.en;
}

export function detectLanguage(): Language {
  return typeof navigator !== 'undefined' && navigator.language.toLowerCase().startsWith('zh')
    ? 'zh'
    : 'en';
}
