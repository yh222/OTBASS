package weka.classifiers.trees;

import java.util.Enumeration;

import weka.attributeSelection.*;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Id3;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.OptionHandler;
import weka.core.Utils;

public class OTBASS extends AbstractClassifier implements OptionHandler {

	private static final long serialVersionUID = -1050710816032117679L;
	ASEvaluation m_evaluator;
	ASSearch m_searcher;
	double m_classValue;
	AttributeSelection m_AttributeSelection = null;
	OTBASS[] m_Successors;
	
	  /** Attribute used for splitting. */
	  private Attribute m_Attribute;

	weka.filters.supervised.attribute.Discretize m_Disc = new weka.filters.supervised.attribute.Discretize();
	weka.filters.supervised.attribute.MergeNominalValues m_Merge = new weka.filters.supervised.attribute.MergeNominalValues();

	@Override
	public void buildClassifier(Instances data) throws Exception {

		makeTree(data);

	}

	private void makeTree(Instances[] splitData) throws Exception {
		for(int i=0;i<splitData.length;i++){
			makeTree(splitData[1]);
		}
	}
	
	private void makeTree(Instances data) throws Exception {

		// Discretize attributes
		Instances discretized = new Instances(data);
		for (Instance instance : data)
			m_Disc.input(instance);
		m_Disc.batchFinished();
		while (m_Disc.outputPeek() != null)
			discretized.add(m_Disc.output());

		// Merge nominal values
		Instances merged = new Instances(discretized);
		for (Instance instance : merged)
			m_Merge.input(instance);
		m_Merge.batchFinished();
		while (m_Merge.outputPeek() != null)
			merged.add(m_Merge.output());

		// if(merged.numAttributes()==1){//create leaf node
		// m_classValue=getClassvalueWithMostCount(merged);
		// }else if(merged.numAttributes()>1){//create successor option node
		//
		// }
		if (noPredictorAttributeHasMoreThanOneValues(merged)) {
			createLeafNode(merged);
		} else {
			// Try to select attribute
			m_AttributeSelection = new AttributeSelection();
			m_AttributeSelection.setEvaluator(m_evaluator);
			m_AttributeSelection.setSearch(m_searcher);
			m_AttributeSelection.SelectAttributes(merged);
			if (m_AttributeSelection.numberAttributesSelected() == 0) {
				createLeafNode(merged);
			} else {
				// Create successor option node
				Instances reduced = m_AttributeSelection
						.reduceDimensionality(merged);
				createDecisionNodes(merged,data);
			}
		}

	}

	private void createDecisionNodes(Instances merged, Instances data) throws Exception {
		m_Successors = new OTBASS[data.numAttributes()];
		
	    Enumeration attEnum = merged.enumerateAttributes();
	    while (attEnum.hasMoreElements()) {
	      Attribute att = (Attribute) attEnum.nextElement();
	      m_Attribute=att;
	      m_Successors[att.index()] = new OTBASS();
	      m_Successors[att.index()].makeTree(splitData(data,att));
	    }
	}



	private void createLeafNode(Instances merged) {
		m_classValue = getClassvalueWithMostCount(merged);
	}

	private boolean noPredictorAttributeHasMoreThanOneValues(Instances merged) {

		return false;
	}

	private double getClassvalueWithMostCount(Instances data) {
		double[] distribution = new double[data.numClasses()];
		Enumeration instEnum = data.enumerateInstances();
		while (instEnum.hasMoreElements()) {
			Instance inst = (Instance) instEnum.nextElement();
			distribution[(int) inst.classValue()]++;
		}
		Utils.normalize(distribution);
		return Utils.maxIndex(distribution);
	}

	  /**
	   * Splits a dataset according to the values of a nominal attribute.
	   *
	   * @param data the data which is to be split
	   * @param att the attribute to be used for splitting
	   * @return the sets of instances produced by the split
	   */
	  private Instances[] splitData(Instances data, Attribute att) {

	    Instances[] splitData = new Instances[att.numValues()];
	    for (int j = 0; j < att.numValues(); j++) {
	      splitData[j] = new Instances(data, data.numInstances());
	    }
	    Enumeration instEnum = data.enumerateInstances();
	    while (instEnum.hasMoreElements()) {
	      Instance inst = (Instance) instEnum.nextElement();
	      splitData[(int) inst.value(att)].add(inst);
	    }
	    for (int i = 0; i < splitData.length; i++) {
	      splitData[i].compactify();
	    }
	    //System.out.println("split data is "+splitData[0].get(0));
	  //  System.out.println(" @seperate by attribute:"+att);
	    //for(Instances ii:splitData){
			//System.out.println(" ");
	    //	printinsts(ii);
	    //}
	    
	    return splitData;
	  }
	
	public void setOptions(String[] options) throws Exception {
		// same for attribute evaluator
		String evaluatorString = Utils.getOption('E', options);
		if (evaluatorString.length() == 0)
			evaluatorString = weka.attributeSelection.CfsSubsetEval.class
					.getName();
		String[] evaluatorSpec = Utils.splitOptions(evaluatorString);
		if (evaluatorSpec.length == 0) {
			throw new Exception(
					"Invalid attribute evaluator specification string");
		}
		String evaluatorName = evaluatorSpec[0];
		evaluatorSpec[0] = "";
		m_evaluator = ASEvaluation.forName(evaluatorName, evaluatorSpec);

		// same for search method
		String searchString = Utils.getOption('S', options);
		if (searchString.length() == 0)
			searchString = weka.attributeSelection.BestFirst.class.getName();
		String[] searchSpec = Utils.splitOptions(searchString);
		if (searchSpec.length == 0) {
			throw new Exception("Invalid search specification string");
		}
		String searchName = searchSpec[0];
		searchSpec[0] = "";
		m_searcher = ASSearch.forName(searchName, searchSpec);

		super.setOptions(options);
	}

	/**
	 * Main method.
	 * 
	 * @param args
	 *            the options for the classifier
	 */
	public static void main(String[] args) {
		runClassifier(new OTBASS(), args);
	}

}
