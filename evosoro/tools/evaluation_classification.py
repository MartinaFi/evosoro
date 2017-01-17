import time
import random
import numpy as np
import subprocess as sub
import re

from ga_classification import classificator

from read_write_voxelyze import read_voxlyze_results, write_voxelyze_file


# TODO: make eval times relative to the number of simulated voxels
# TODO: right now just saving files gen-id-fitness; but this should be more flexible (as option in objective dict?)
# TODO: fitness isn't even necessarily the name of the top objective --> use pop.objective_dict[0]["name"] (?)
# getattr(ind, pop.objective_dict[0]["name"])
# TODO: the location of voxelyze and the data must be consistent and specified or more robust (cp for now)
# sub.call("cp ../_voxcad/voxelyzeMain/voxelyze .", shell=True)
# sub.call("cp ../_voxcad/qhull .", shell=True)


def evaluate_all(sim, env, pop, print_log, save_vxa_every, run_directory, run_name, max_eval_time=60,
                 time_to_try_again=10, save_lineages=False):
    """Evaluate all individuals of the population in VoxCad.

    Parameters
    ----------
    sim : Sim
        Configures parameters of the Voxelyze simulation.

    env : Env
        Configures parameters of the Voxelyze environment.

    pop : Population
        This provides the individuals to evaluate.

    print_log : PrintLog()
        For logging with time stamps

    save_vxa_every : int
        Which generations to save information about individual SoftBots

    run_directory : string
        Where to save

    run_name : string
        Experiment name for files

    max_eval_time : int
        How long to run physical simulation per ind in pop

    time_to_try_again : int
        How long to wait until relaunching remaining unevaluated (crashed?) simulations

    save_lineages : bool
        Save the vxa of every ancestor of the surviving individual

    """
    start_time = time.time()
    num_evaluated_this_gen = 0
    ids_to_analyze = []
    re_sel_id_scen = re.compile("softbotsOutput--id_([0-9]+)--scen_([0-9]+).xml")
    evals_done = {}

    for ind in pop:

        ind.md5 = {};
        for scenario in sim.scenarios:

            # write the phenotype of a SoftBot to a file so that VoxCad can access for sim.
            scen_md5  = write_voxelyze_file(sim, env, ind, run_directory, run_name, scenario)
            ind.md5[scenario['scen_id']] = scen_md5
            # don't evaluate if invalid
            if not ind.phenotype.is_valid():
                for rank, goal in pop.objective_dict.items():
                    if goal["name"] != "age":
                        setattr(ind, goal["name"], goal["worst_value"])
                print_log.message("Skipping invalid individual")

            # don't evaluate if identical phenotype has already been evaluated
            elif env.actuation_variance == 0 and scen_md5 in pop.already_evaluated:
                for rank, goal in pop.objective_dict.items():
                    if goal["tag"] is not None:
                        setattr(ind, goal["name"], pop.already_evaluated[scen_md5][rank])
                        print_log.message("Individual already evaluated:  cached fitness is {}".format(ind.fitness))

                if pop.gen % save_vxa_every == 0 and save_vxa_every > 0:
                    sub.call("cp " + run_directory + "/voxelyzeFiles/" + run_name + "--id_%05i--scen_%05i.vxa" % (ind.id, scenario['scen_id']) +
                             " " + run_directory + "/Gen_%04i/" % pop.gen + run_name +
                             "--Gen_%04i--fit_%.08f--id_%05i--scen_%05i.vxa" % (pop.gen, ind.fitness, ind.id, scenario['scen_id']), shell=True)

            # otherwise evaluate with voxelyze
            else:
                num_evaluated_this_gen += 1
                pop.total_evaluations += 1
                ids_to_analyze += ["--id_%05i--scen_%05i" % (ind.id, scenario['scen_id'])]

                sub.Popen("./voxelyze  -f " + run_directory + "/voxelyzeFiles/" + run_name + "--id_%05i--scen_%05i.vxa" % (ind.id, scenario['scen_id']) ,
                          shell=True)

    print_log.message("Launched {0} voxelyze calls, out of {1} individuals".format(num_evaluated_this_gen, len(pop)))

    num_evals_finished = 0
    all_done = False
    already_analyzed_ids = []
    redo_attempts = 1

    fitness_eval_start_time = time.time()

    while not all_done:

        time_waiting_for_fitness = time.time() - fitness_eval_start_time
        # this protects against getting stuck when voxelyze doesn't return a fitness value
        # (diverging simulations, crash, error reading .vxa)

        if time_waiting_for_fitness > pop.pop_size * max_eval_time:
            # TODO ** WARNING: This could in fact alter the sim and undermine the reproducibility **
            all_done = False  # something bad with this individual, probably sim diverged
            break

        if time_waiting_for_fitness > pop.pop_size * time_to_try_again * redo_attempts:
            # try to redo any simulations that crashed
            redo_attempts += 1
            non_analyzed_ids = [idx for idx in ids_to_analyze if idx not in already_analyzed_ids]
            print "Rerunning voxelyze for: ", non_analyzed_ids
            for idx in non_analyzed_ids:
                sub.Popen("./voxelyze  -f " + run_directory + "/voxelyzeFiles/" + run_name + idx + ".vxa",
                          shell=True)

        # check to see if all are finished
        all_done = True
        sbagliati = 0
        for ind_id, evaluated_list in evals_done.items():
            if evaluated_list.__len__() < sim.scenarios.__len__():
                all_done = False
                sbagliati +=1
        print "Sbagliati: %d" % sbagliati

        # check for any fitness files that are present
        ls_check = sub.check_output(["ls", run_directory + "/fitnessFiles/"])
        # duplicated ids issue: may be due to entering here two times for the same fitness file found in the directory.

        if ls_check:
            # ls_check = random.choice(ls_check.split())  # doesn't accomplish anything and undermines reproducibility
            ls_check = ls_check.split()[0]
            if "softbotsOutput--id_" in ls_check:
                mtc = re_sel_id_scen.match(ls_check)
                this_id = "--id_%05i--scen_%05i" % (int(mtc.group(1)), int(mtc.group(2)))
                ind_id = mtc.group(1)
                int_id_as_num = int(ind_id)
                scenario = (item for item in sim.scenarios if item["scen_id"] == int(mtc.group(2))).next() # Looing for the right scenario

                if this_id in already_analyzed_ids:
                    # workaround to avoid any duplicated ids when restarting sims
                    print_log.message("Duplicate voxelyze results found from THIS gen with id {}".format(this_id))
                    sub.call("rm " + run_directory + "/fitnessFiles/" + ls_check, shell=True)

                elif this_id in pop.all_evaluated_individuals_ids:
                    print_log.message("Duplicate voxelyze results found from PREVIOUS gen with id {}".format(this_id))
                    sub.call("rm " + run_directory + "/fitnessFiles/" + ls_check, shell=True)

                else:
                    num_evals_finished += 1
                    already_analyzed_ids.append(this_id)

                    ind_filename = run_directory + "/fitnessFiles/" + ls_check
                    objective_values_dict = read_voxlyze_results(pop, print_log, ind_filename)

                    print_log.message("{0} fit = {1} ({2} / {3})".format(ls_check, objective_values_dict[0],
                                                                         num_evals_finished,
                                                                         num_evaluated_this_gen))

                    # now that we've read the fitness file, we can remove it
                    sub.call("rm " + run_directory + "/fitnessFiles/" + ls_check, shell=True)

                    # assign the values to the corresponding individual
                    for ind in pop:
                        if ind.id == int_id_as_num:
                            evaluated_values = {}
                            for rank, details in pop.objective_dict.items():
                                if objective_values_dict[rank] is not None:
                                    if not details["node_func"]:
                                        evaluated_values[details["name"]] = objective_values_dict[rank]
                                    else:
                                        for name, details_phenotype in ind.genotype.to_phenotype_mapping.items():
                                            if name == details["output_node_name"]:
                                                state = details_phenotype["state"]
                                                evaluated_values[details["name"]] = details["node_func"](state, objective_values_dict[rank])
                                else:
                                    for name, details_phenotype in ind.genotype.to_phenotype_mapping.items():
                                        if name == details["output_node_name"]:
                                            state = details_phenotype["state"]
                                            evaluated_values[details["name"]] = details["node_func"](state)

                                if pop.objective_dict[rank]['mode'] == "classification":
                                    evaluated_values[details["name"]] = {"value": evaluated_values[details["name"]], "target": scenario["target"]}

                            pop.already_evaluated[ind.md5[scenario['scen_id']]] = [getattr(ind, details["name"])
                                                              for rank, details in
                                                              pop.objective_dict.items()]
                            pop.all_evaluated_individuals_ids += [this_id]


                            if evals_done.get(ind_id, None) is None:
                                evals_done[ind_id] = [evaluated_values]
                            else:
                                evals_done[ind_id] += [evaluated_values]

                            break

            # wait a second and try again
            else:
                time.sleep(0.5)
        else:
            time.sleep(0.5)

    if not all_done:
        print_log.message("WARNING: Couldn't get a fitness value in time for some individuals, "
                          "probably there's something wrong with them (e.g. sim diverged). "
                          "The min fitness is being assigned to those")

    for ind_id, evaluated_list in evals_done.items():
        calculated_values = {}
        class_values = []
        num = evaluated_list.__len__()
        class_obj = (item for item in pop.objective_dict.items() if item["mode"] == "classification").next()
        ind =  (item for item in pop if item["id"] == ind_id).next()

        for evaluated in evaluated_list:
            for name, value in evaluated:
                if name == class_obj.name:
                    class_values += [{
                        value: evaluated['value'],
                        target: evaluated['target']
                    }]
                else:
                    calculated_values[name] = calculated_values[name] + 1/num * value

        calculated_values[class_obj.name] = classificator(class_values);
        for name, value in calculated_values:
            setattr(ind, name, value)
        # update the run statistics and file management
        if ind.fitness > pop.best_fit_so_far:
            pop.best_fit_so_far = ind.fitness

            for scenario in sim.scenarios:
                my_id = "--id_%05i--scen_%05i" % (ind.id, scenario['scen_id'])
                sub.call("cp " + run_directory + "/voxelyzeFiles/" + run_name + my_id +".vxa" +
                        " " + run_directory + "/bestSoFar/fitOnly/" + run_name +
                         "--Gen_%04i--fit_%.08f" % (pop.gen, ind.fitness)
                         + my_id + ".vxa", shell=True)

        if save_lineages:
            for scenario in sim.scenarios:
                my_id = "--id_%05i--scen_%05i" % (ind.id, scenario['scen_id'])
                sub.call("cp " + run_directory + "/voxelyzeFiles/" + run_name + my_id +".vxa" +
                         " " + run_directory + "/lineages/" +
                         run_name + "--Gen_%04i--fit_%.08f" % (pop.gen, ind.fitness) +
                         my_id + " .vxa", shell=True)

        if pop.gen % save_vxa_every == 0 and save_vxa_every > 0:
            for scenario in sim.scenarios:
                my_id = "--id_%05i--scen_%05i" % (ind.id, scenario['scen_id'])
                sub.call("mv " + run_directory + "/voxelyzeFiles/" + run_name + my_id +".vxa" +
                         " " + run_directory + "/Gen_%04i/" % pop.gen +
                         run_name + "--Gen_%04i--fit_%.08f" % (pop.gen, ind.fitness) +
                         my_id + ".vxa", shell=True)
        else:
            for scenario in sim.scenarios:
                my_id = "--id_%05i--scen_%05i" % (ind.id, scenario['scen_id'])
                sub.call("rm " + run_directory + "/voxelyzeFiles/" + run_name + my_id +
                ".vxa", shell=True)

    print_log.message("\nAll Voxelyze evals finished in {} seconds".format(time.time() - start_time))
    print_log.message("num_evaluated_this_gen: {0}".format(num_evaluated_this_gen))
    print_log.message("total_evaluations: {}".format(pop.total_evaluations))
