library(stagedtrees)

estimate_joint_distribution <- function(data){
  cardinalities <- data[1,]
    # Generate all the outcomes in the order specified by the data labels
  spaces <- list()
  i <- 1
  for (c in cardinalities) {
    spaces[[colnames(data)[[i]]]] <- seq(c)-1
    i <- i + 1
  }
  space <- rev(expand.grid(rev(spaces)))
  
  ### search the order 
  data <- data[-1,] ## remove first row
  model <- search_best(data, alg = stages_bhc)
  
  ## generate the full sample space
  prob <- prob(model, space)
  log_prob <- prob(model, space, log=TRUE)
  df <- cbind(space, prob = prob, log_prob=log_prob)
  
}

samp_size_range <- c(1000)
seeds <-  seq(10)-1
num_levels_range <- c(5, 7)
path <- "sim_results"


for (num_levels in num_levels_range){
  for (samp_size in samp_size_range){
    for (seed in seeds){
      set.seed(seed)
      name <- paste0("p=",num_levels,"_n=",samp_size,"_seed=",seed,".csv")
      data <- read.csv(paste0(path, "/data/", name), check.names = FALSE)      
      est_path <- paste0(path, "/distr/stagedtrees/", name)      
      print(est_path)
        # if file exists, skip
        if (file.exists(est_path)){
            next
        }
      dir.create(paste0(path, "/distr/stagedtrees/"))
      df <- estimate_joint_distribution(data)
      write.csv(df, file = est_path, row.names = FALSE, quote = FALSE)
    }
  }
}

estimate_joint_distribution(data)
