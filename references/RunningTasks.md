# Writing on the Wall

## Writing
### Literature Review
- Find a more generalized resource on wave modelling
- maybe try to find a recent book
- [ ] `https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=ocean+wave+swell+modelling+survey&btnG=`
- start writing the literature review in an actual structured way
- transition to the LaTex Document
 - [Laugel et al.(2014)] Laugel, Amélie and Menendez, Melisa and Benoit, Michel and Mattarolo, Giovanni and
Méndez, Fernando. (2014) Wave climate projections along the French coastline: dynamical versus statistical
downscaling methods, Ocean Modelling. 48, 35–50.
 - [Wang et al.(2010)] Wang, Xiaolan L and Swail, Val R and Cox, Andrew. (2010) Dynamical versus statistical
downscaling methods for ocean wave he
 - [Obakrim et al.(2022)] Obakrim, Said and Ailliot, Pierre and Monbet, Valerie and Raillard, Nicolas. (2022)
Statistical mode

### Unofficial Lit Review
 - https://paperswithcode.com/area/time-series 

## Coding
- come up with function inversion for the newly defined index representation 
- [ ] Originally the data update stream comes in every 30 min => 48 times a day
- [ ] After averaging the data, we want 2 data points per day ( ~ one for morning and one for evening)
- [ ] So we need to average the data in approximately 12 hour chunks
- [ ] We can do this by taking the average of the first 24 observations, then the next 24 observations, etc.
- [ ] This will give us 2 data points per day
- [ ] This amounts to ~730 observations per year, instead of ~17520 (approximations to account for missing data)
- [ ] In reverse order for the index representation, to ensure well positioned test region  
- remove the outlier
- do the averaging
- try the exact same analysis we've been doing but on a smaller set (e.g. last 1000 or 5000 observations)  

## Notes
### Wind 
- [ ] Onshore winds (blows from the ocean towards the land) generally create poor surf conditions
- [ ] Offshore winds (blows from the land out to sea) generally create better surf conditions
- [ ] What is ocean wave downscaling (downsampling?) and ocean upscaling (upsampling?))

