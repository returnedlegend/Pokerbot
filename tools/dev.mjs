import http from 'node:http';
import { spawn } from 'node:child_process';
import { existsSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { setTimeout as delay } from 'node:timers/promises';

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const TITLE = '<title>Play jr007 · Poker table</title>';

export function options(args = process.argv.slice(2), env = process.env) {
  const result = { host: env.HOST || '0.0.0.0', port: env.PORT || 3000,
    backendPort: env.POKER_BACKEND_PORT || 8080 };
  for (let i = 0; i < args.length; i++) {
    if (!['--host', '--port'].includes(args[i]) || !args[i + 1]) {
      throw new Error('Usage: npm run dev -- [--host 0.0.0.0] [--port 3000]');
    }
    result[args[i].slice(2)] = args[++i];
  }
  for (const name of ['port', 'backendPort']) {
    result[name] = Number(result[name]);
    if (!Number.isInteger(result[name]) || result[name] < 1 || result[name] > 65535) {
      throw new Error(`${name} must be a port from 1 to 65535`);
    }
  }
  if (result.port === result.backendPort) throw new Error('Frontend and backend ports must differ');
  return result;
}

export function pythonCommand(root = ROOT, env = process.env) {
  if (env.POKER_PYTHON) return env.POKER_PYTHON;
  for (const suffix of ['.venv/bin/python', '.venv/Scripts/python.exe']) {
    const path = resolve(root, suffix);
    if (existsSync(path)) return path;
  }
  return process.platform === 'win32' ? 'python' : 'python3';
}

export function probeTable(port) {
  // A static-page request neither deals a hand nor creates a player session.
  return new Promise((accept, reject) => {
    const request = http.get({ hostname: '127.0.0.1', port, path: '/' }, response => {
      let html = '';
      response.setEncoding('utf8');
      response.on('data', chunk => {
        html += chunk;
        if (html.length > 256 * 1024) request.destroy(new Error('Unexpected service response'));
      });
      response.on('end', () => {
        if (response.statusCode === 200 && html.includes(TITLE)) accept(true);
        else reject(new Error(`Port ${port} is occupied by another service. Set POKER_BACKEND_PORT to a free port.`));
      });
      response.on('error', reject);
    });
    request.setTimeout(2000, () => request.destroy(new Error(`Backend on port ${port} did not respond`)));
    request.on('error', error => error.code === 'ECONNREFUSED' ? accept(false) : reject(error));
  });
}

export function createProxy(backendPort) {
  return http.createServer((request, response) => {
    const upstream = http.request({ hostname: '127.0.0.1', port: backendPort,
      method: request.method, path: request.url,
      headers: { ...request.headers, host: `127.0.0.1:${backendPort}`,
        'x-forwarded-host': request.headers.host || '',
        'x-forwarded-proto': request.headers['x-forwarded-proto'] || 'http' } }, incoming => {
      response.writeHead(incoming.statusCode, incoming.headers);
      incoming.pipe(response);
      incoming.on('error', () => response.destroy());
    });
    upstream.setTimeout(30000, () => upstream.destroy(new Error('Backend request timed out')));
    upstream.on('error', () => {
      if (!response.headersSent) {
        response.writeHead(502, { 'Content-Type': 'application/json', 'Cache-Control': 'no-store' });
        response.end(JSON.stringify({ detail: 'The poker engine disconnected. Restart npm run dev to reconnect.' }));
      } else response.destroy();
    });
    request.on('aborted', () => upstream.destroy());
    request.pipe(upstream); // Never retry a wager or share state between browsers.
  });
}

export async function main() {
  const config = options();
  let child = null, server = null, stopping = false, childFailure = null;
  async function shutdown() {
    if (stopping) return;
    stopping = true;
    if (server) {
      server.closeAllConnections();
      await new Promise(done => server.close(done));
    }
    // This reference is set only when this launcher starts the engine.
    if (child?.pid && child.exitCode === null && child.signalCode === null) {
      const exited = new Promise(done => child.once('exit', done));
      child.kill('SIGTERM');
      const timer = setTimeout(() => child.kill('SIGKILL'), 5000);
      timer.unref();
      await exited;
      clearTimeout(timer);
    }
  }
  const interrupted = () => shutdown().then(() => process.exit(0));
  process.once('SIGINT', interrupted);
  process.once('SIGTERM', interrupted);
  try {
    if (!await probeTable(config.backendPort)) {
      console.log('Starting the Python poker engine…');
      child = spawn(pythonCommand(), [resolve(ROOT, 'tools/human_play.py'),
        '--host', '127.0.0.1', '--port', String(config.backendPort)], { cwd: ROOT, stdio: 'inherit',
        env: { ...process.env, OPENBLAS_NUM_THREADS: '1', OMP_NUM_THREADS: '1',
          MKL_NUM_THREADS: '1', NUMBA_NUM_THREADS: '1' } });
      child.once('error', error => { childFailure = error; });
      child.once('exit', (code, signal) => {
        if (!stopping) {
          childFailure = new Error(`Python engine exited (${signal || code})`);
          if (server?.listening) {
            console.error(childFailure.message);
            shutdown().then(() => { process.exitCode = 1; });
          }
        }
      });
      const deadline = Date.now() + 45000;
      while (!await probeTable(config.backendPort)) {
        if (childFailure) throw childFailure;
        if (Date.now() >= deadline) throw new Error('Python engine did not start within 45 seconds');
        await delay(200);
      }
    } else console.log(`Using your existing poker engine on port ${config.backendPort}.`);
    server = createProxy(config.backendPort);
    await new Promise((accept, reject) => {
      server.once('error', reject);
      server.listen(config.port, config.host, accept);
    });
    console.log(`\nPlay jr007: http://localhost:${config.port}\nPress Ctrl+C to stop the web table.\n`);
  } catch (error) {
    console.error(`\n${error.message}\nPython setup: python3 -m venv .venv, then .venv/bin/pip install -r requirements.txt\n`);
    await shutdown();
    process.exitCode = 1;
  }
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) await main();
