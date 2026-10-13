library(tidyverse)
library(lubridate)
library(stringr)

# read in data
env <- read.csv("data/all_env_data_2025.csv")
fluor <- read.csv("data/all_fluorescence_2025.csv")
tree <- read.csv("data/big_tree_data.csv")
canopy <- read.csv("data/dscmet_421_b_5min_2025.csv")
water <- read.csv("data/hja_wp_dat.csv")

##Fluorescence data set
fluor <- fluor %>%
  mutate(start_date = mdy(start_date),
         needle_age = needle.age,
         tree_id = as.factor(tree_id),
         measurement = as.factor(measurement, levels=c("initial", "0","1","7","9","14")),
         panel_id = as.factor(panel_id),
         tree_treatment = as.factor(tree_treatment)) 

head(fluor)

tree_info <- tree %>%
  mutate(start_date = mdy(start_date),
         measurement_date = mdy(measurement_date),
         dayPPFD = (average_day_lux_HOBO * 0.0185),
         maxPPFD = (maximum_lux_HOBO * 0.0185)) %>%
  select(start_date,tree_id,panel_id,tree_treatment,heat_treatment,needle_age,measurement,measurement_date, dayPPFD, maxPPFD)

end_dates <- tree_info %>%
  filter(measurement == 0) %>%
  distinct(tree_id, panel_id, end_date = measurement_date)

tree_info <- tree_info %>%
  left_join(end_dates, by = c("tree_id", "panel_id"))

all_flu <- merge(fluor, tree_info, by = c("start_date","tree_id","tree_treatment","panel_id","needle_age","measurement","heat_treatment"))

all_flu <- all_flu %>%
  mutate(collect_time = ymd_hm(paste(measurement_date, "09:00"),
                               tz = "America/Los_Angeles"))
head(all_flu)


#Water potential data set
head(water)
water <- water %>%
  filter(spp == "Pseudotsuga menziesii") %>%
  mutate(date= dmy(paste(date, 2025)), 
         tree_treatment = case_when(treatment == "n" ~ "control", treatment == "i" ~ "irrigated"),
         predawn_time  = ymd_hm(paste(date, "04:00"), tz = "America/Los_Angeles"),
         midday_time   = ymd_hm(paste(date, "13:00"), tz = "America/Los_Angeles")) %>%
  rename(tree_id = id) %>%
  select(-treatment, -spp, -key)

head(water)

wp <- water %>%
  pivot_longer(c(predawn, midday),
               names_to = "wp_type", values_to = "wp_value") %>%
  mutate(collect_time = if_else(wp_type == "predawn", predawn_time, midday_time)) %>%
  select(-predawn_time, -midday_time)

head(wp)


#Met station data
canopy <- canopy[-c(1:2),] 
canopy <- canopy %>%
  subset(select = c("Date","WSPD_SNC_MAX_150_0_01","WSPD_SNC_MEAN_150_0_01"))

env <- left_join(env, canopy, by = "Date")

head(env)

env_clean <- env %>%
  mutate(Date_time = mdy_hm(Date),        
         Date_only = as.Date(Date_time),   
         airtemp = (AIRTEMP_MEAN_150_0_04), 
         windspeed = (WSPD_SNC_MEAN_150_0_01), 
         vpd_kpa = (VPD_MEAN_150_0_04 * 0.1)) %>%
  select(Date_time, Date_only, airtemp, vpd_kpa, windspeed)

head(env_clean)
