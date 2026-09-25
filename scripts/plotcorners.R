# plot corners from champions league group stage.
library(tidyverse)
library(dplyr)
library(StatsBombR) # Our Data
library(ggsoccer) # Pitch Map
Comps <- FreeCompetitions()
Comps <- Comps %>%
  filter(competition_name == "UEFA Women's Euro" & season_name == "2025")
Matches <- FreeMatches(Comps)

England <- Matches %>%
  filter(home_team.home_team_name == "England Women's" | away_team.away_team_name == "England Women's") # Filtering for England Women's

StatsBombData <- free_allevents(MatchesDF = England, Parallel = T)
StatsBombData = allclean(StatsBombData)
xG <- StatsBombData %>%
  select(id = shot.key_pass_id, xG = shot.statsbomb_xg)
inswinging_left_passes <- StatsBombData %>%
  filter(pass.type.name == "Corner") %>%
  mutate(total_corners_left = n(), # Total Corners Column
         short = sum(is.na(pass.technique.name)), # Short Corners
         created_shot = ifelse((pass.shot_assist == TRUE | pass.goal_assist == TRUE), TRUE, NA)) %>%
  filter(pass.technique.name == "Inswinging" & location.y < 40 & (pass.end_location.x > 102 & pass.end_location.y > 18 & pass.end_location.y < 62)) %>%
  left_join(xG) %>%
  mutate(total_inswinging = n(), 
         pct_total = n()/total_corners_left,
         xG = sum(xG, na.rm = TRUE),
         shots = sum((pass.goal_assist == TRUE | pass.shot_assist == TRUE), na.rm = TRUE ),
         location.y = 80 - location.y, # Adjusting y Coordinate
         ave_end_x = (mean(pass.end_location.x)), # Average x Coordinate
         ave_end_y = 80 - (mean(pass.end_location.y))) # Average y Coordinate
