"""research_objective_v1: explicit opt-in composition for the corrected baseline."""
import math
from nca.losses import LossSpec,mean_terms,FAMILIES
from nca.interventions import objective_terms
from nca.facade import facade_term,facade_bounds
from nca.regularizers import density_binary,total_variation,cantilever_boundary
VERSION='research_objective_v1'
REGULARIZERS=('density_binary','tv','cantilever_boundary')


def research_terms(state,raw,context,config,allowance,spec=LossSpec()):
    """Fixed experimental contract selected by D028; no legacy defaults changed."""
    result=objective_terms(state,raw,context,config,'hard_preclamp','envelope',spec)
    p=state[:,config['ch_structure']]
    result['terms']['facade']=facade_term(p,context,allowance,spec)
    joint=facade_bounds(context,'envelope',allowance,spec)
    result['context_valid']=result['context_valid'] & joint['joint_necessary_compatible']
    result['joint_bounds']=joint
    result['regularizers']={'density_binary':density_binary(p),'tv':total_variation(p),
                            'cantilever_boundary':cantilever_boundary(p,context.support)}
    result['objective_version']=VERSION
    return result


def weighted_total(result,family_weights,regularizer_weights):
    """Require a complete declared recipe; no omitted families or hidden defaults.

    All nine family coefficients stay positive. Retained regularizers may have
    zero coefficients for explicit ablations; old cantilever cannot be mixed in.
    """
    if set(family_weights)!=set(FAMILIES) or set(regularizer_weights)!=set(REGULARIZERS):
        raise ValueError('Recipe must name exactly all nine families and three regularizers')
    for name,value in {**family_weights,**regularizer_weights}.items():
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value<0 or (name in FAMILIES and value==0):
            raise ValueError('Invalid explicit coefficient: '+name)
    terms=mean_terms(result)
    return sum(terms[n]*family_weights[n] for n in FAMILIES)+sum(result['regularizers'][n].mean()*regularizer_weights[n] for n in REGULARIZERS)
