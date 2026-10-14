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
         measurement = as.factor(measurement),
         panel_id = as.factor(panel_id),
         tree_treatment = as.factor(tree_treatment)) 

head(fluor)

tree_info <- tree %>%
  mutate(start_date = mdy(start_date),
         measurement_date = mdy(measurement_date),
         dayPPFD = (average_day_lux_HOBO * 0.0185),
         maxPPFD = (maximum_lux_HOBO * 0.0185)) %>%
  select(measurement_date,tree_id,panel_id,tree_treatment,heat_treatment,needle_age,measurement,dayPPFD, maxPPFD, start_date)

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

#Full data set
env_clean <- env_clean %>%
  mutate(windspeed = as.numeric(windspeed))

env_sum <- env_clean %>%
  group_by(Date_only) %>%
  summarise(
    max_airtemp    = max(airtemp, na.rm = TRUE),
    mean_airtemp   = mean(airtemp, na.rm = TRUE),
    min_airtemp    = min(airtemp, na.rm = TRUE),
    max_vpd        = max(vpd_kpa, na.rm = TRUE),
    mean_vpd       = mean(vpd_kpa, na.rm = TRUE),
    min_vpd        = min(vpd_kpa, na.rm = TRUE),
    mean_windspeed = mean(windspeed, na.rm = TRUE),
    .groups = "drop") %>%
  filter(Date_only > "2025-04-30")

head(env_sum)
head(water)

joined <- env_sum %>%
  left_join(water, by = c("Date_only" = "date"))

all_join <- joined %>%
  left_join(all_flu, by = c("Date_only" = "measurement_date","tree_id","tree_treatment"))

head(all_join)

adjusted <- all_join %>%
  mutate(
    max_airtemp_adj = if_else(
      heat_treatment == "heated" & Date_only >= start_date & Date_only < end_date,
      max_airtemp + 2,
      max_airtemp),
    mean_airtemp_adj = if_else(
      heat_treatment == "heated" & Date_only >= start_date & Date_only < end_date,
      mean_airtemp + 2,
      mean_airtemp))

names(adjusted)

SAM_data <- adjusted %>%
  rename(date = Date_only,
         panel_on = start_date,
         panel_off = end_date,
         FvFm = Fv.Fm) %>%
  select(-predawn_time, -midday_time, -collect_time, -needle.age)

names(SAM_data)

SAM_data <- SAM_data[, c(1,2,24,3,25,4,5,6,7,8,9,12,15,14,17,21,22,13,23,16,10,11,18,19,20)]

write.csv(SAM_data, "data/SAM_data.csv", row.names=FALSE)
