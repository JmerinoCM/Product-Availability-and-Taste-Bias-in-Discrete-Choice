################################################################################
# PONAE: Probability of a number of alternatives exhausted as the choice unfolds
# Graph generation
################################################################################

rm(list=ls())
# Libraries
library(readr)
library(ggplot2)
library(ggstream)
library(viridis)
library(tidyverse)
library(cowplot)

# Data graphing function

crear_grafico <- function(J_name, categoria_name, q_name) {
  
  # Data preparation
  main_directory <- "C:/Projects/ProductAvaility/Repository"
  
  datos <- read_csv(paste0(main_directory,"/",
                           categoria_name,"/",
                           "J",J_name,"/",
                           "prob_nJ/",
                           "prob_n-J",J_name,"_q",q_name,".csv"))
  
  prob_n_df <- datos[, -1]  # Exclude the probability that no product will run out of stock
  colnames(prob_n_df) <- as.numeric(gsub("P_", "", colnames(prob_n_df)))  

  prob_n_df_long <- prob_n_df %>%
    mutate(Tiempo = 1:n()) %>%
    pivot_longer(cols = -Tiempo, names_to = "Grupo", values_to = "Valor")
  
  prob_n_df_long$Grupo <- factor(prob_n_df_long$Grupo, 
                                      levels = rev(sort(as.numeric(unique(prob_n_df_long$Grupo)))))

  suma_cum <- prob_n_df_long %>% 
    group_by(Tiempo) %>%
    summarise(
      Valor = sum(Valor)  
    ) %>%
    mutate(
      Grupo = factor(999)  
    ) 
  
  datos_adt <- prob_n_df_long %>%
    mutate(n_obs = 1:n()) %>%
    filter(Valor != 0)
  
  datos_adt$Grupo <- as.numeric(as.character(datos_adt$Grupo))
  
  if (nrow(datos_adt) == 0) {
    min_row <- data.frame(Grupo = 0)
    max_row <- data.frame(Grupo = 0)
  } else {
    min_row <- datos_adt[datos_adt$n_obs ==  min(datos_adt$n_obs), ]
    max_row <- datos_adt[datos_adt$n_obs ==  max(datos_adt$n_obs), ]
  }
  
  # Plot
  g_p <- ggplot(prob_n_df_long, aes(x = Tiempo, y = Valor, fill = Grupo)) +
    geom_area(position = "stack",
              alpha = 1,
              color = 0,
              lwd = 0.05) +  
    scale_fill_grey(breaks = as.character(as.numeric(min_row$Grupo):as.numeric(max_row$Grupo)),
                    start = 0.9,
                    end = 0.1) + 
    theme_minimal() +
    labs(x = expression(n),
         y = "Probability",
         fill = "Alternatives\nExhausted\n(Number)",
         title = "") + 
    theme(
      legend.key.width = unit(0.25, "cm"),  
      legend.key.height = unit(0.25, "cm"),  
      legend.text = element_text(size = 10), 
      legend.box.margin = margin(0, 0, 0, 0, "cm"),
      legend.margin = margin(0, 0, 0, 0, "cm"),
      legend.box.just = "center",
      legend.title = element_text(size = 12),
      legend.position = "bottom",
      legend.box = "vertical",
      axis.title.x = element_text(size = 12),
      axis.title.y = element_text(size = 12)
    ) +
    theme(
      panel.grid.major = element_blank(), 
      panel.grid.minor = element_blank(), 
      strip.background = element_blank(), 
      strip.text = element_text(size = 10, face = "bold"),
      axis.text.x = element_text(angle = 45, hjust = 1) 
    ) +
    theme(panel.spacing = unit(.05, "lines"),
          panel.border = element_rect(color = "gray50", fill = NA, size = 1), 
          strip.background = element_rect(color = "gray50", size = 1)) +
    scale_x_continuous(
      limits = c(1, 100),  
      breaks = c(1, seq(10, 100, by = 10))  
    ) +
    geom_vline(xintercept = c(1, seq(10, 100, by = 10)), linetype = "dotted", color = "black") +
    geom_hline(yintercept = 0.5, linetype = "dotted", color = "black") + 
    geom_hline(yintercept = 1, linetype = "dotted", color = "black") + 
    geom_hline(yintercept = 0.0, linetype = "dotted", color = "black") 
  
  g_p <- g_p +
    geom_line(data = suma_cum,
              aes(x = Tiempo, y = Valor, group = 1,
                  color = "simpleID"),
              size = 1.5,
              linetype = "solid") +
    scale_color_manual(
      values = c("simpleID" = "black"),
      labels = expression(P(J[n] != J ~ "|" ~ beta * "," ~ J * "," ~ q * "," ~ n)),
      name = " "
    ) +
    guides(color = guide_legend(title = expression(P(J[n] != J ~ "|" ~ beta * "," ~ J * "," ~ q * "," ~ n)))) +
    guides(fill = guide_legend(ncol = 15, byrow = TRUE, reverse = F, order = 2),
                   colour = guide_legend(order = 1))
  
  output_folder <- file.path(paste(main_directory,"graphs_capacity_n",sep="/"))
  
  if (!dir.exists(output_folder)) {
    dir.create(output_folder)
  }
  
  output_plot <- paste0(output_folder,"/",
                        categoria_name,
                        "_J=", J_name,
                        "_q=", q_name,
                        ".pdf")
  ggsave(output_plot, plot = g_p, width = 20, height = 9.5, units = "cm") 
  
}

# Run function

## Setup
valores_J <- c(100, 50, 10, 5)

valores_q <- list(
  c(1 ,  2,  3,  5, 20, 50, 100),
  c(2 ,  3,  4,  5, 20, 50, 100),
  c(10, 11, 12, 15, 20, 50, 100),
  c(20, 21, 22, 25, 40, 50, 100)
)

degree_differentiation <- c("homog", "heterog")

for (j in valores_J) {
  for (type_degree in degree_differentiation) {
    for (q in valores_q[[which(valores_J == j)]]) {

      resultado <- crear_grafico(as.character(j), type_degree, as.character(q))
      
    }
  }
}
