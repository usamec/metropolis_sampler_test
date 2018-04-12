#[macro_use]
extern crate ndarray;
extern crate rand;

// example usage:
use rand::{Rng, thread_rng};
use rand::RngCore;
use rand::distributions::{Normal, Distribution};

use ndarray::{Array2, Array, Array1};
use ndarray::Zip;
use ndarray::FoldWhile;

const CONTROL: usize = 11;
const SICK: usize = 6;
const NEXP: usize = 30;

fn str_to_array(s: &str, r: usize, c: usize) -> Array2<f64> {
    let raw_vec = s.split('\n').flat_map(|line| {
        line.split(' ').map(|x| x.parse().unwrap())
    }).collect::<Vec<f64>>();
    Array::from_shape_vec((r, c), raw_vec).unwrap()
}

fn load_data() -> (Array2<f64>, Array2<f64>) {
    let str_control = "312 272 350 286 268 328 298 356 292 308 296 372 396 402 280 330 254 282 350 328 332 308 292 258 340 242 306 328 294 272
354 346 384 342 302 312 322 376 306 402 320 298 308 414 304 422 388 422 426 338 332 426 478 372 392 374 430 388 354 368
256 284 320 274 324 268 370 430 314 312 362 256 342 388 302 366 298 396 274 226 328 274 258 220 236 272 322 284 274 356
260 294 306 292 264 290 272 268 344 362 330 280 354 320 334 276 418 288 338 350 350 324 286 322 280 256 218 256 220 356
204 272 250 260 314 308 246 236 208 268 272 264 308 236 238 350 272 252 252 236 306 238 350 206 260 280 274 318 268 210
590 312 286 310 778 364 318 316 316 298 344 262 274 330 312 310 376 326 346 334 282 292 282 300 290 302 300 306 294 444
308 364 374 278 366 310 358 380 294 334 302 250 542 340 352 322 372 348 460 322 374 370 334 360 318 356 338 346 462 510
244 240 278 262 266 254 240 244 226 266 294 250 284 260 418 280 294 216 308 324 264 232 294 236 226 234 274 258 208 380
232 262 230 222 210 284 232 228 264 246 264 316 260 266 304 268 384 234 308 266 294 254 222 262 278 290 208 232 206 206
318 324 282 364 286 342 306 302 280 306 256 334 332 336 360 344 480 310 336 314 392 284 292 280 320 322 286 406 352 324
240 292 350 254 396 430 260 320 298 312 290 248 276 364 318 434 400 382 318 298 298 248 250 234 280 306 282 234 424 244";
    let str_sick = "276 272 264 258 278 286 314 340 334 364 286 344 312 380 262 324 310 260 280 262 364 316 270 286 326 302 300 302 344 290
374 466 432 376 360 454 478 382 524 410 520 470 514 354 434 380 416 384 462 386 404 362 420 360 390 356 550 372 386 396
594 1014 1586 1344 610 838 772 264 748 1076 446 314 304 1680 1700 334 256 422 302 296 354 322 276 382 502 428 544 286 650 432
402 466 296 348 680 702 500 500 576 624 406 378 586 826 298 882 564 656 716 380 448 506 1714 748 510 810 984 458 390 642
620 714 414 358 460 598 324 442 372 410 998 636 968 490 696 560 562 720 618 456 502 974 1032 470 462 798 716 300 586 574
454 388 344 226 562 766 502 432 608 516 500 796 542 458 448 404 372 524 400 366 374 350 1154 558 440 348 400 460 514 450";

    (str_to_array(str_control, 11, 30), str_to_array(str_sick, 6, 30))
}

#[derive(Clone)]
struct Model {
    mu: f64,
    delta_alphas_ctrl: Array1<f64>,
    delta_alphas_sick: Array1<f64>,
    tau: f64,
    sigma_ctrl: f64,
    sigma_sick: f64,
    lambdas: Array1<f64>,
    z: Array2<f64>
}

fn normal_density(x: f64, mean: f64, stdev: f64) -> f64 {
    return -((x - mean)*(x - mean)) / (2.0 * stdev * stdev) + (1.0 / stdev).ln()
}

fn half_normal_density(x: f64, stdev: f64) -> f64 {
    return -(x*x) / (2.0 * stdev * stdev) + (1.0 / stdev).ln()
}

impl Model {
    fn new() -> Model {
        let mut rng = thread_rng();
        let normal = Normal::new(0.0, 50.0);
        Model {
            mu : 400.0,
            delta_alphas_ctrl: Array1::from_shape_fn((CONTROL), |_| {
                normal.sample(&mut rng) as f64
            }),
            delta_alphas_sick: Array1::from_shape_fn((SICK), |_| {
                normal.sample(&mut rng) as f64
            }),
            tau: 50.0,
            sigma_ctrl: 100.0,
            sigma_sick: 100.0,
            lambdas : Array1::from_shape_fn((SICK), |_| 0.5),
            z: Array2::from_shape_fn((SICK, NEXP), |_| (rng.next_u32() % 2) as f64)
        }
    }

    fn log_prior(&self) -> f64 {
        normal_density(self.mu, 400.0, 100.0) +
            self.delta_alphas_ctrl.iter().map(|&x| normal_density(x, 0.0, 50.0)).sum::<f64>() +
            self.delta_alphas_sick.iter().map(|&x| normal_density(x, 0.0, 50.0)).sum::<f64>() +
            half_normal_density(self.tau, 100.0) +
            half_normal_density(self.sigma_ctrl, 100.0) +
            half_normal_density(self.sigma_sick, 100.0) +
            // no prior for lambdas needed
            Zip::from(self.z.genrows()).and(&self.lambdas).fold_while(0.0, |acc, zrow, lambda| {
                let ones = zrow.scalar_sum();
                let zeros = zrow.len() as f64 - ones;
                FoldWhile::Continue(acc + lambda.ln() * ones + (1.0 - lambda).ln() * zeros)
            }).into_inner()
    }

    fn log_data(&self, data_control: &Array2<f64>, data_sick: &Array2<f64>) -> f64 {
        let ctrl = Zip::from(data_control.genrows()).and(&self.delta_alphas_ctrl).fold_while(0.0, |acc, data, delta| {
           FoldWhile::Continue(acc + data.fold(0.0, |acc, &measurement| {
               acc + normal_density(measurement, self.mu + delta, self.sigma_ctrl)
           }))
        }).into_inner();
        let sick = Zip::from(data_sick.genrows()).and(self.z.genrows()).and(&self.delta_alphas_sick).fold_while(0.0, |acc, data, zrow, delta| {
           FoldWhile::Continue(acc + Zip::from(data).and(zrow).fold_while(0.0, |acc_inner, &measurement, &z| {
               let log_p = if z < 0.01 {
                   normal_density(measurement, self.mu + delta, self.sigma_ctrl)
               } else {
                   normal_density(measurement, self.mu + delta + self.tau, self.sigma_sick)
               };
               FoldWhile::Continue(acc_inner + log_p)
           }).into_inner())
        }).into_inner();
        ctrl + sick
    }

    fn log_posterior(&self, data_control: &Array2<f64>, data_sick: &Array2<f64>) -> f64 {
        self.log_prior() + self.log_data(data_control, data_sick)
    }

    fn propose(&self) -> Model {
        let normal1 = Normal::new(0.0, 1.0);
        Model {
            mu: self.mu + normal1.sample(&mut thread_rng()) * 1.0,
            delta_alphas_ctrl: self.delta_alphas_ctrl.mapv(|delta| {
                delta + normal1.sample(&mut thread_rng()) * 2.0
            }),
            delta_alphas_sick: self.delta_alphas_sick.mapv(|delta| {
                delta + normal1.sample(&mut thread_rng()) * 2.0
            }),
            tau: (self.tau + normal1.sample(&mut thread_rng()) * 2.0).abs(),
            sigma_ctrl: (self.sigma_ctrl + normal1.sample(&mut thread_rng()) * 2.0).abs(),
            sigma_sick: (self.sigma_sick + normal1.sample(&mut thread_rng()) * 2.0).abs(),
            lambdas : self.lambdas.mapv(|lambda| {
                let mut new_lambda = lambda + normal1.sample(&mut thread_rng()) * 0.003;
                while new_lambda < 0.0 || new_lambda > 1.0 {
                    if new_lambda < 0.0 {
                        new_lambda = -new_lambda
                    }
                    if new_lambda > 1.0 {
                        new_lambda = 1.0 - new_lambda
                    }
                }
                new_lambda
            }),
            z: self.z.mapv(|z| {
                if thread_rng().next_u32() % 100 == 0 {
                    1.0 - z
                } else {
                    z
                }
            })
        }
    }
}

fn summary<F>(chain: &Vec<Model>, pick: F, skip: usize, name: &str) where F: Fn(&Model) -> f64 {
    let valid_samples = chain.len() - skip;
    let (sum, sum2) = chain.iter().skip(skip).fold((0.0, 0.0), |(sum, sum2), model| {
        let val = pick(model);
        (sum + val, sum2 + val*val)
    });
    let mean = sum / valid_samples as f64;
    println!("{} {:.2}+-{:.2}", name, mean, (sum2 / valid_samples as f64 - mean * mean).sqrt());
}

fn main() {
    let (data_control, data_sick) = load_data();

    let mut model = Model::new();

    let mut old_posterior = model.log_posterior(&data_control, &data_sick);
    let mut chain = Vec::new();
    let mut tries = 0u64;
    let total_samples = 1000000;
    let skip = 100000;
    while chain.len() < total_samples {
        tries += 1;
        let new_model = model.propose();
        let new_posterior = new_model.log_posterior(&data_control, &data_sick);
        if new_posterior > old_posterior || thread_rng().gen::<f64>() < (new_posterior - old_posterior).exp(){
            model = new_model;
            old_posterior = new_posterior;
            println!("acc {} {} {} sctrl {} ssick {} mu {} tau {} l {:?}", chain.len(), tries, new_posterior, model.sigma_ctrl, model.sigma_sick, model.mu, model.tau, model.lambdas);
            chain.push(model.clone());
        }
    }

    summary(&chain, |x| x.mu, skip, "mu");
    summary(&chain, |x| x.tau, skip, "tau");
    summary(&chain, |x| x.sigma_ctrl, skip, "sigma_ctrl");
    summary(&chain, |x| x.sigma_sick, skip, "sigma_sick");
    for i in 0..6 {
        summary(&chain, |x| x.lambdas[i], skip, &format!("lambda_{}", i));
    }
}
