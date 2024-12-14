import config from '../config';
import { ApiClientInterface } from './types';
import { routes, Route } from './routes';

export class ApiClientRest implements ApiClientInterface {
    [key: string]: any;

    constructor() {
        this.load_modules();
    }

    private load_modules() {
        routes.forEach((route: Route) => {
            if (route.module.create && typeof route.module.create === 'function') {
                const instance = route.module.create(this);
                if (route.path === '') {
                    // Top-level module
                    Object.assign(this, instance);
                } else {
                    this[route.path] = instance;
                }
            }
        });
    }

    private async request<T>(
        method: string,
        path: string,
        on_error: (messages: string[]) => void,
        data?: any
    ): Promise<T | null> {
        console.log("Request:", config.api_base_url, path);
        const full_url = new URL(path, config.api_base_url).toString();
        const headers: Record<string, string> = {
            'Accept': 'application/json',
        };
        const options: RequestInit = {
            method,
            headers,
            credentials: 'include' as const
        };
        if (method !== 'GET' && data) {
            headers['Content-Type'] = 'application/json';
            options.body = JSON.stringify(data);
        }
        try {
            const response = await fetch(full_url, options);
            if (response.status === 401) {
                const error_data = await response.json();
                on_error([error_data.detail || 'Not authenticated']);
                throw new Error('Not authenticated');
            }
            if (response.status === 403) {
                const error_data = await response.json();
                on_error([error_data.detail || 'Not authorized']);
                throw new Error('Not authorized');
            }
            if (response.status === 204) {
                return null;
            }
            if (!response.ok) {
                const error_data = await response.json();
                on_error([error_data.detail || `HTTP error! status: ${response.status}`]);
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            return await response.json();
        } catch (error) {
            console.error("Request error:", error);
            throw error;
        }
    }

    public get<T>(path: string, on_error: (messages: string[]) => void): Promise<T> {
        return this.request<T>('GET', path, on_error) as Promise<T>;
    }

    public post<T>(path: string, on_error: (messages: string[]) => void, data: any): Promise<T> {
        return this.request<T>('POST', path, on_error, data) as Promise<T>;
    }

    public put<T>(path: string, on_error: (messages: string[]) => void, data: any): Promise<T> {
        return this.request<T>('PUT', path, on_error, data) as Promise<T>;
    }

    public delete(path: string, on_error: (messages: string[]) => void): Promise<null> {
        return this.request<null>('DELETE', path, on_error);
    }
}
