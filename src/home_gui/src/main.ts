import { mount } from 'svelte';
import App from './App.svelte';
import './app.css';

document.addEventListener('submit', (event) => event.preventDefault(), true);

mount(App, { target: document.getElementById('app')! });
