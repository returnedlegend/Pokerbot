import assert from 'node:assert/strict';
import http from 'node:http';
import { test } from 'node:test';
import { options, createProxy, probeTable } from './dev.mjs';

async function listen(server) {
  await new Promise(done => server.listen(0, '127.0.0.1', done));
  return server.address().port;
}
async function close(server) {
  server.closeAllConnections();
  await new Promise(done => server.close(done));
}

test('configuration accepts normal npm arguments and rejects collisions/bad ports', () => {
  assert.deepEqual(options(['--host', '127.0.0.1', '--port', '3011'], {}),
    { host: '127.0.0.1', port: 3011, backendPort: 8080 });
  assert.equal(options([], { PORT: '3009', POKER_BACKEND_PORT: '8099' }).backendPort, 8099);
  assert.throws(() => options([], { PORT: '8080' }), /must differ/);
  assert.throws(() => options(['--port', 'bad'], {}), /port/);
});

test('readiness inspects only the static page and rejects another service', async () => {
  const requests = [];
  const good = http.createServer((request, response) => {
    requests.push(request.url);
    response.end('<html><title>Play jr007 · Poker table</title></html>');
  });
  const bad = http.createServer((_, response) => response.end('unrelated service'));
  try {
    assert.equal(await probeTable(await listen(good)), true);
    assert.deepEqual(requests, ['/']);
    await assert.rejects(probeTable(await listen(bad)), /another service/);
  } finally { await close(good); await close(bad); }
});

test('proxy forwards one POST, private cookies, status and Set-Cookie exactly', async () => {
  const received = [];
  const backend = http.createServer((request, response) => {
    let body = '';
    request.on('data', chunk => { body += chunk; });
    request.on('end', () => {
      received.push({ method: request.method, url: request.url, cookie: request.headers.cookie, body });
      response.writeHead(409, { 'Set-Cookie': ['session=changed; HttpOnly; SameSite=Strict'],
        'Content-Type': 'application/json' });
      response.end(JSON.stringify({ detail: 'stale request' }));
    });
  });
  let proxy;
  try {
    const backendPort = await listen(backend); proxy = createProxy(backendPort);
    const port = await listen(proxy), payload = JSON.stringify({ action: 'RAISE_POT', request_id: 'one' });
    const result = await new Promise((done, reject) => {
      const request = http.request({ host: '127.0.0.1', port, path: '/api/action', method: 'POST',
        headers: { Cookie: 'session=private-one', 'Content-Type': 'application/json' } }, response => {
        let body = ''; response.on('data', chunk => { body += chunk; });
        response.on('end', () => done({ status: response.statusCode, cookie: response.headers['set-cookie'], body }));
      }); request.on('error', reject); request.end(payload);
    });
    assert.equal(result.status, 409);
    assert.deepEqual(result.cookie, ['session=changed; HttpOnly; SameSite=Strict']);
    assert.equal(JSON.parse(result.body).detail, 'stale request');
    assert.deepEqual(received, [{ method: 'POST', url: '/api/action', cookie: 'session=private-one', body: payload }]);
  } finally { if (proxy) await close(proxy); await close(backend); }
});
