# Footway Hackathon Cillers demo
This project serves as a starting point for the Cillers Footway Hackathon.

It's a simple demo app with building blocks in place for you to start building your own personal shopper app using the OpenAI or Anthropic and Footway APIs.

## Prerequisites
To get started, make sure you install Polytope:

### Docker or OrbStack
You'll need Docker or OrbStack to run the app. You can install Docker from [here](https://docs.docker.com/get-docker/) and OrbStack from [here](https://docs.orbstack.dev/install).

### Polytope CLI
On macOS:
```bash
brew install polytopelabs/tap/polytope-cli
```

Also make sure you're running at least Polytope 0.1.31:
```bash
polytope --version
- The current CLI version is: 0.1.31-bae4935de-macos-arm64
```

If you're on an older version, you can update it with:
```bash
brew upgrade polytope-cli
```

For installation on Windows and Linux, see [the docs](https://polytope.com/docs/quick-start).

## Components
This app has two main components:
- [The API](./code/api) - A simple REST API written in Python that serves as a backend for the app.
- [The UI](./web-app) - A simple React UI written in TypeScript that serves as a frontend for the app.

## Running the app
To run the app, clone this repository and navigate to the project directory:

```bash
git clone https://github.com/aeriksson/cillers-footway-hackathon.git
cd cillers-footway-hackathon
```

Then run the following command to start the app:
```
pt run stack
```

Then open the UI at [http://localhost:3000](http://localhost:3000). This can take a little while to return something useful - especially on the first run, because it needs to set up all the dependencies.

API documentation is automatically generated and can be found at [http://localhost:3001/docs](http://localhost:3001/docs).

You'll have hot reload, so any changes you make to the code will be reflected in the UI immediately - however, if you add or remove packages you'll need to restart the app.
