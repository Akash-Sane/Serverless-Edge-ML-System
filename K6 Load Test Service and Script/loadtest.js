import http from 'k6/http';
import { sleep, check } from 'k6';

const COLD_START_THRESHOLD = 1000; // Define the threshold for cold starts (in ms)
const WARM_START_THRESHOLD = 500;  // Define the threshold for warm starts (in ms)

export let options = {
  stages: [
    { duration: '1m', target: 20 },   // ramp up to 20 users
    { duration: '1m', target: 20 },   // stay at 20 users
    { duration: '1m', target: 50 },   // ramp up to 40 users
    { duration: '1m', target: 50 },   // stay at 40 users
    { duration: '1m', target: 100 },   // ramp up to 80 users
    { duration: '1m', target: 100 },   // stay at 80 users
    { duration: '1m', target: 300 },  // ramp up to 120 users
    { duration: '1m', target: 300 },  // stay at 120 users
    { duration: '1m', target: 900 },  // ramp up to 160 users
    { duration: '1m', target: 900 },  // stay at 160 users
    { duration: '1m', target: 1400 },  // ramp up to 200 users
    { duration: '1m', target: 1400 },  // stay at 200 users
    { duration: '1m', target: 1600 },  // ramp up to 200 users
    { duration: '1m', target: 1600 },  // stay at 200 users
    { duration: '1m', target: 1800 },  // ramp up to 200 users
    { duration: '1m', target: 1800 },  // stay at 200 users
    { duration: '1m', target: 2000 },  // ramp up to 200 users
    { duration: '1m', target: 2000 },  // stay at 200 users
    { duration: '1m', target: 2100 },  // ramp up to 200 users
    { duration: '1m', target: 2100 },  // stay at 200 users
    { duration: '1m', target: 0 }     // ramp down to 0 users
  ],
  thresholds: {
    'http_req_duration': ['p(95)<500'], // 95% of requests should be below 500ms
  },
};

const urls = [
  'http://preprocess-and-train-mnist.default.172.18.1.101.sslip.io/health',
  'http://preprocess-and-train-imagenet.default.172.18.1.101.sslip.io/health',
  'http://preprocess-and-train-cifar10.default.172.18.1.101.sslip.io/health',
  'http://predict.default.172.18.1.101.sslip.io/health',
  'http://model-manager.default.172.18.1.101.sslip.io/health'
];

export default function () {
  let randomIndex = Math.floor(Math.random() * urls.length);
  let res = http.get(urls[randomIndex]);

  // Detecting potential cold and warm starts and logging them
  if (res.timings.duration > COLD_START_THRESHOLD) {
    console.log(`Cold Start: ${res.timings.duration}ms for URL: ${urls[randomIndex]}`);
  } else if (res.timings.duration <= WARM_START_THRESHOLD) {
    console.log(`Warm Start: ${res.timings.duration}ms for URL: ${urls[randomIndex]}`);
  }

  // Check the response
  check(res, {
    'status was 200': (r) => r.status === 200,
    'response time was acceptable': (r) => r.timings.duration < 500
  });

  sleep(1);
}
