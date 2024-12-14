import * as top_level_module from './modules/top_level';

// Add more module imports here as needed

export interface Route {
    path: string;
    module: any; // You might want to create a more specific type for modules
}

export const routes: Route[] = [
    { path: '', module: top_level_module }, // Empty string for top-level routes
    // Add more routes here as needed
];
