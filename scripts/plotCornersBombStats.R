library(tidyverse)
library(dplyr)
library(StatsBombR) # Our Data
library(ggsoccer) # Pitch Map

# Get competitions
Comps <- FreeCompetitions()

# LA LIGA 2010/2011 SEASON
Comps_2010 <- Comps %>%
  filter(competition_name == "La Liga" & season_name == "2010/2011")

Matches_2010 <- FreeMatches(Comps_2010)

# Filter for ONLY El Clásico matches
Clasicos <- Matches_2010 %>%
  filter((home_team.home_team_name == "Barcelona" & away_team.away_team_name == "Real Madrid") |
           (home_team.home_team_name == "Real Madrid" & away_team.away_team_name == "Barcelona"))

# Check how many Clásicos we have
print(paste("Number of Clásicos found:", nrow(Clasicos)))
View(Clasicos)

# Get all events from Clásicos
StatsBombData_clasicos <- free_allevents(MatchesDF = Clasicos, Parallel = T)
StatsBombData_clasicos <- allclean(StatsBombData_clasicos)

# Create xG lookup
xG_clasicos <- StatsBombData_clasicos %>%
  select(id = shot.key_pass_id, xG = shot.statsbomb_xg)

# ============= BARCELONA IN CLASICOS =============
# BARCELONA LEFT
inswinging_left_barca <- StatsBombData_clasicos %>%
  filter(team.name == "Barcelona" & pass.type.name == "Corner") %>%
  mutate(total_corners_left = n(),
         short = sum(is.na(pass.technique.name)),
         created_shot = ifelse((pass.shot_assist == TRUE | pass.goal_assist == TRUE), TRUE, NA)) %>%
  filter(pass.technique.name == "Inswinging" & location.y < 40 & 
           (pass.end_location.x > 102 & pass.end_location.y > 18 & pass.end_location.y < 62)) %>%
  left_join(xG_clasicos, by = "id") %>%
  mutate(total_inswinging = n(), 
         pct_total = n()/total_corners_left,
         xG = sum(xG, na.rm = TRUE),
         shots = sum((pass.goal_assist == TRUE | pass.shot_assist == TRUE), na.rm = TRUE),
         location.y = 80 - location.y,
         ave_end_x = mean(pass.end_location.x),
         ave_end_y = 80 - mean(pass.end_location.y))

# BARCELONA RIGHT
inswinging_right_barca <- StatsBombData_clasicos %>%
  filter(team.name == "Barcelona" & pass.type.name == "Corner") %>%
  mutate(total_corners_right = n(),
         short = sum(is.na(pass.technique.name)),
         created_shot = ifelse((pass.shot_assist == TRUE | pass.goal_assist == TRUE), TRUE, NA)) %>%
  filter(pass.technique.name == "Inswinging" & location.y > 40 & 
           (pass.end_location.x > 102 & pass.end_location.y > 18 & pass.end_location.y < 62)) %>%
  left_join(xG_clasicos, by = "id") %>%
  mutate(total_inswinging = n(), 
         pct_total = n()/total_corners_right,
         xG = sum(xG, na.rm = TRUE),
         shots = sum((pass.goal_assist == TRUE | pass.shot_assist == TRUE), na.rm = TRUE),
         location.y = 80 - location.y,
         ave_end_x = mean(pass.end_location.x),
         ave_end_y = 80 - mean(pass.end_location.y))

# ============= REAL MADRID IN CLASICOS =============
# REAL MADRID LEFT
inswinging_left_madrid <- StatsBombData_clasicos %>%
  filter(team.name == "Real Madrid" & pass.type.name == "Corner") %>%
  mutate(total_corners_left = n(),
         short = sum(is.na(pass.technique.name)),
         created_shot = ifelse((pass.shot_assist == TRUE | pass.goal_assist == TRUE), TRUE, NA)) %>%
  filter(pass.technique.name == "Inswinging" & location.y < 40 & 
           (pass.end_location.x > 102 & pass.end_location.y > 18 & pass.end_location.y < 62)) %>%
  left_join(xG_clasicos, by = "id") %>%
  mutate(total_inswinging = n(), 
         pct_total = n()/total_corners_left,
         xG = sum(xG, na.rm = TRUE),
         shots = sum((pass.goal_assist == TRUE | pass.shot_assist == TRUE), na.rm = TRUE),
         location.y = 80 - location.y,
         ave_end_x = mean(pass.end_location.x),
         ave_end_y = 80 - mean(pass.end_location.y))

# REAL MADRID RIGHT
inswinging_right_madrid <- StatsBombData_clasicos %>%
  filter(team.name == "Real Madrid" & pass.type.name == "Corner") %>%
  mutate(total_corners_right = n(),
         short = sum(is.na(pass.technique.name)),
         created_shot = ifelse((pass.shot_assist == TRUE | pass.goal_assist == TRUE), TRUE, NA)) %>%
  filter(pass.technique.name == "Inswinging" & location.y > 40 & 
           (pass.end_location.x > 102 & pass.end_location.y > 18 & pass.end_location.y < 62)) %>%
  left_join(xG_clasicos, by = "id") %>%
  mutate(total_inswinging = n(), 
         pct_total = n()/total_corners_right,
         xG = sum(xG, na.rm = TRUE),
         shots = sum((pass.goal_assist == TRUE | pass.shot_assist == TRUE), na.rm = TRUE),
         location.y = 80 - location.y,
         ave_end_x = mean(pass.end_location.x),
         ave_end_y = 80 - mean(pass.end_location.y))

# ============= PLOTS - BARCELONA IN CLASICOS =============
plot_barca_left <- ggplot() +
  annotate_pitch(dimensions = pitch_statsbomb, fill = "gray15") +
  theme_pitch() +
  coord_flip(xlim = c(90, 125), ylim = c(80, 0)) +
  geom_curve(data = inswinging_left_barca,
             aes(x = location.x, xend = pass.end_location.x, y = location.y, yend = pass.end_location.y),
             colour = "white", alpha = 0.9, arrow = arrow(length = unit(0.2, "cm"), type = "closed"), 
             linewidth = 0.8, curvature = 0.2, angle = 90) +
  geom_curve(data = inswinging_left_barca, 
             aes(x = location.x, xend = ave_end_x, y = location.y, yend = ave_end_y),
             colour = "#A50044", alpha = 1, arrow = arrow(length = unit(0.2, "cm"), type = "closed"), 
             linewidth = 1.5, curvature = 0.2, angle = 90) +
  geom_point(aes(x = 101, y = 56), shape = 21, fill = "white", colour = "#A50044", size = 18) +
  geom_text(data = inswinging_left_barca, aes(x = 101, y = 56, label = scales::percent(round(pct_total, 2))),
            colour = "black", size = 5, family = "sans", check_overlap = TRUE) +
  geom_text(aes(x = 93, y = 56, label = "Usage %"), colour = "white", size = 5, family = "sans", check_overlap = TRUE) +
  geom_point(aes(x = 101, y = 40), shape = 21, fill = "#A50044", colour = "#A50044", size = 18) +
  geom_text(data = inswinging_left_barca, aes(x = 101, y = 40, label = round(xG, 2)),
            colour = "white", size = 5, family = "sans", check_overlap = TRUE) +
  geom_text(aes(x = 93, y = 40, label = "xG"), colour = "white", size = 5, family = "sans", check_overlap = TRUE) +
  geom_point(aes(x = 101, y = 24), shape = 21, fill = "white", colour = "#A50044", size = 18) +
  geom_text(data = inswinging_left_barca, aes(x = 101, y = 24, label = shots),
            colour = "black", size = 5, family = "sans", check_overlap = TRUE) +
  geom_text(aes(x = 93, y = 24, label = "Shots"), colour = "white", size = 5, family = "sans", check_overlap = TRUE) +
  theme(plot.title = element_text(family = "sans", face = "bold", size = 16, colour = "white"),
        plot.subtitle = element_text(family = "sans", colour = "white"),
        plot.caption = element_text(family = "sans", colour = "white"),
        plot.background = element_rect(fill = "gray15"),
        panel.background = element_rect(fill = "gray15"),
        aspect.ratio = c(60/100),
        plot.margin = margin(15, 15, 10, 15)) +
  labs(title = "Barcelona Left Inswinging Corners - El Clásico",
       subtitle = "La Liga 2010/2011 | Head-to-Head Matches Only",
       caption = "Carr Analytics | thefalsenine.substack.com")

plot_barca_right <- ggplot() +
  annotate_pitch(dimensions = pitch_statsbomb, fill = "gray15") +
  theme_pitch() +
  coord_flip(xlim = c(90, 125), ylim = c(80, 0)) +
  geom_curve(data = inswinging_right_barca,
             aes(x = location.x, xend = pass.end_location.x, y = location.y, yend = pass.end_location.y),
             colour = "white", alpha = 0.9, arrow = arrow(length = unit(0.2, "cm"), type = "closed"), 
             linewidth = 0.8, curvature = 0.2, angle = 90) +
  geom_curve(data = inswinging_right_barca, 
             aes(x = location.x, xend = ave_end_x, y = location.y, yend = ave_end_y),
             colour = "#EDBB00", alpha = 1, arrow = arrow(length = unit(0.2, "cm"), type = "closed"), 
             linewidth = 1.5, curvature = 0.2, angle = 90) +
  geom_point(aes(x = 101, y = 56), shape = 21, fill = "white", colour = "#EDBB00", size = 18) +
  geom_text(data = inswinging_right_barca, aes(x = 101, y = 56, label = scales::percent(round(pct_total, 2))),
            colour = "black", size = 5, family = "sans", check_overlap = TRUE) +
  geom_text(aes(x = 93, y = 56, label = "Usage %"), colour = "white", size = 5, family = "sans", check_overlap = TRUE) +
  geom_point(aes(x = 101, y = 40), shape = 21, fill = "#EDBB00", colour = "#EDBB00", size = 18) +
  geom_text(data = inswinging_right_barca, aes(x = 101, y = 40, label = round(xG, 2)),
            colour = "black", size = 5, family = "sans", check_overlap = TRUE) +
  geom_text(aes(x = 93, y = 40, label = "xG"), colour = "white", size = 5, family = "sans", check_overlap = TRUE) +
  geom_point(aes(x = 101, y = 24), shape = 21, fill = "white", colour = "#EDBB00", size = 18) +
  geom_text(data = inswinging_right_barca, aes(x = 101, y = 24, label = shots),
            colour = "black", size = 5, family = "sans", check_overlap = TRUE) +
  geom_text(aes(x = 93, y = 24, label = "Shots"), colour = "white", size = 5, family = "sans", check_overlap = TRUE) +
  theme(plot.title = element_text(family = "sans", face = "bold", size = 16, colour = "white"),
        plot.subtitle = element_text(family = "sans", colour = "white"),
        plot.caption = element_text(family = "sans", colour = "white"),
        plot.background = element_rect(fill = "gray15"),
        panel.background = element_rect(fill = "gray15"),
        aspect.ratio = c(60/100),
        plot.margin = margin(15, 15, 10, 15)) +
  labs(title = "Barcelona Right Inswinging Corners - El Clásico",
       subtitle = "La Liga 2010/2011 | Head-to-Head Matches Only",
       caption = "Carr Analytics | thefalsenine.substack.com")

# ============= PLOTS - REAL MADRID IN CLASICOS =============
plot_madrid_left <- ggplot() +
  annotate_pitch(dimensions = pitch_statsbomb, fill = "gray15") +
  theme_pitch() +
  coord_flip(xlim = c(90, 125), ylim = c(80, 0)) +
  geom_curve(data = inswinging_left_madrid,
             aes(x = location.x, xend = pass.end_location.x, y = location.y, yend = pass.end_location.y),
             colour = "white", alpha = 0.9, arrow = arrow(length = unit(0.2, "cm"), type = "closed"), 
             linewidth = 0.8, curvature = 0.2, angle = 90) +
  geom_curve(data = inswinging_left_madrid, 
             aes(x = location.x, xend = ave_end_x, y = location.y, yend = ave_end_y),
             colour = "#FEBE10", alpha = 1, arrow = arrow(length = unit(0.2, "cm"), type = "closed"), 
             linewidth = 1.5, curvature = 0.2, angle = 90) +
  geom_point(aes(x = 101, y = 56), shape = 21, fill = "white", colour = "#FEBE10", size = 18) +
  geom_text(data = inswinging_left_madrid, aes(x = 101, y = 56, label = scales::percent(round(pct_total, 2))),
            colour = "black", size = 5, family = "sans", check_overlap = TRUE) +
  geom_text(aes(x = 93, y = 56, label = "Usage %"), colour = "white", size = 5, family = "sans", check_overlap = TRUE) +
  geom_point(aes(x = 101, y = 40), shape = 21, fill = "#FEBE10", colour = "#FEBE10", size = 18) +
  geom_text(data = inswinging_left_madrid, aes(x = 101, y = 40, label = round(xG, 2)),
            colour = "black", size = 5, family = "sans", check_overlap = TRUE) +
  geom_text(aes(x = 93, y = 40, label = "xG"), colour = "white", size = 5, family = "sans", check_overlap = TRUE) +
  geom_point(aes(x = 101, y = 24), shape = 21, fill = "white", colour = "#FEBE10", size = 18) +
  geom_text(data = inswinging_left_madrid, aes(x = 101, y = 24, label = shots),
            colour = "black", size = 5, family = "sans", check_overlap = TRUE) +
  geom_text(aes(x = 93, y = 24, label = "Shots"), colour = "white", size = 5, family = "sans", check_overlap = TRUE) +
  theme(plot.title = element_text(family = "sans", face = "bold", size = 16, colour = "white"),
        plot.subtitle = element_text(family = "sans", colour = "white"),
        plot.caption = element_text(family = "sans", colour = "white"),
        plot.background = element_rect(fill = "gray15"),
        panel.background = element_rect(fill = "gray15"),
        aspect.ratio = c(60/100),
        plot.margin = margin(15, 15, 10, 15)) +
  labs(title = "Real Madrid Left Inswinging Corners - El Clásico",
       subtitle = "La Liga 2010/2011 | Head-to-Head Matches Only",
       caption = "Carr Analytics | thefalsenine.substack.com")

plot_madrid_right <- ggplot() +
  annotate_pitch(dimensions = pitch_statsbomb, fill = "gray15") +
  theme_pitch() +
  coord_flip(xlim = c(90, 125), ylim = c(80, 0)) +
  geom_curve(data = inswinging_right_madrid,
             aes(x = location.x, xend = pass.end_location.x, y = location.y, yend = pass.end_location.y),
             colour = "white", alpha = 0.9, arrow = arrow(length = unit(0.2, "cm"), type = "closed"), 
             linewidth = 0.8, curvature = 0.2, angle = 90) +
  geom_curve(data = inswinging_right_madrid, 
             aes(x = location.x, xend = ave_end_x, y = location.y, yend = ave_end_y),
             colour = "#00529F", alpha = 1, arrow = arrow(length = unit(0.2, "cm"), type = "closed"), 
             linewidth = 1.5, curvature = 0.2, angle = 90) +
  geom_point(aes(x = 101, y = 56), shape = 21, fill = "white", colour = "#00529F", size = 18) +
  geom_text(data = inswinging_right_madrid, aes(x = 101, y = 56, label = scales::percent(round(pct_total, 2))),
            colour = "black", size = 5, family = "sans", check_overlap = TRUE) +
  geom_text(aes(x = 93, y = 56, label = "Usage %"), colour = "white", size = 5, family = "sans", check_overlap = TRUE) +
  geom_point(aes(x = 101, y = 40), shape = 21, fill = "#00529F", colour = "#00529F", size = 18) +
  geom_text(data = inswinging_right_madrid, aes(x = 101, y = 40, label = round(xG, 2)),
            colour = "white", size = 5, family = "sans", check_overlap = TRUE) +
  geom_text(aes(x = 93, y = 40, label = "xG"), colour = "white", size = 5, family = "sans", check_overlap = TRUE) +
  geom_point(aes(x = 101, y = 24), shape = 21, fill = "white", colour = "#00529F", size = 18) +
  geom_text(data = inswinging_right_madrid, aes(x = 101, y = 24, label = shots),
            colour = "black", size = 5, family = "sans", check_overlap = TRUE) +
  geom_text(aes(x = 93, y = 24, label = "Shots"), colour = "white", size = 5, family = "sans", check_overlap = TRUE) +
  theme(plot.title = element_text(family = "sans", face = "bold", size = 16, colour = "white"),
        plot.subtitle = element_text(family = "sans", colour = "white"),
        plot.caption = element_text(family = "sans", colour = "white"),
        plot.background = element_rect(fill = "gray15"),
        panel.background = element_rect(fill = "gray15"),
        aspect.ratio = c(60/100),
        plot.margin = margin(15, 15, 10, 15)) +
  labs(title = "Real Madrid Right Inswinging Corners - El Clásico",
       subtitle = "La Liga 2010/2011 | Head-to-Head Matches Only",
       caption = "Carr Analytics | thefalsenine.substack.com")

# Save all plots
ggsave("clasico_barca_left.png", plot = plot_barca_left, width = 10, height = 8, dpi = 300)
ggsave("clasico_barca_right.png", plot = plot_barca_right, width = 10, height = 8, dpi = 300)
ggsave("clasico_madrid_left.png", plot = plot_madrid_left, width = 10, height = 8, dpi = 300)
ggsave("clasico_madrid_right.png", plot = plot_madrid_right, width = 10, height = 8, dpi = 300)

# Display plots
plot_barca_left
plot_barca_right
plot_madrid_left
plot_madrid_right
