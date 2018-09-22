/* Open source system for classification learning from very large data
 ** Copyright (C) 2012 Geoffrey I Webb
 **
 ** This program is free software: you can redistribute it and/or modify
 ** it under the terms of the GNU General Public License as published by
 ** the Free Software Foundation, either version 3 of the License, or
 ** (at your option) any later version.
 **
 ** This program is distributed in the hope that it will be useful,
 ** but WITHOUT ANY WARRANTY; without even the implied warranty of
 ** MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 ** GNU General Public License for more details.
 **
 ** You should have received a copy of the GNU General Public License
 ** along with this program. If not, see <http://www.gnu.org/licenses/>.
 **
 ** Please report any bugs to Geoff Webb <geoff.webb@monash.edu>
 */
#include "ltanas.h"
#include "utils.h"
#include "correlationMeasures.h"
#include <assert.h>
#include <math.h>
#include <set>
#include <stdlib.h>
#include <iostream>
ltanas::ltanas() :
trainingIsFinished_(false)
{
}

ltanas::ltanas(char* const *&, char* const *) :
xxyDist_(), trainingIsFinished_(false)
{
    name_ = "tanas";
}
class miCmpClass
{
public:

    miCmpClass(std::vector<float> *m)
    {
        mi = m;
    }

    bool operator()(CategoricalAttribute a, CategoricalAttribute b)
    {
        return (*mi)[a] > (*mi)[b];
    }

private:
    std::vector<float> *mi;
};
ltanas::~ltanas(void)
{
}

void ltanas::reset(InstanceStream &is)
{
    instanceStream_ = &is;
    const unsigned int noCatAtts = is.getNoCatAtts();
    noCatAtts_ = noCatAtts;
    noClasses_ = is.getNoClasses();

    trainingIsFinished_ = false;

    //safeAlloc(parents, noCatAtts_);
    parents_.resize(noCatAtts);
    tanparents_.resize(noCatAtts);
    for (CategoricalAttribute a = 0; a < noCatAtts_; a++)
    {
        parents_[a] = NOPARENT;
        tanparents_[a]=NOPARENT;
    }

    xxyDist_.reset(is);
}

void ltanas::getCapabilities(capabilities &c)
{
    c.setCatAtts(true); // only categorical attributes are supported at the moment
}

void ltanas::initialisePass()
{
    assert(trainingIsFinished_ == false);
    //learner::initialisePass (pass_);
    //	dist->clear();
    //	for (CategoricalAttribute a = 0; a < meta->noCatAtts; a++) {
    //		parents_[a] = NOPARENT;
    //	}
}

void ltanas::train(const instance &inst)
{
    xxyDist_.update(inst);
}

void ltanas::classify(const instance &inst, std::vector<double> &classDist)
{
    //printf("nbbbb");
    std::vector<double> tanclassDist;
    tanclassDist.resize(noClasses_);
    for (CategoricalAttribute a = 0; a < noCatAtts_; a++)
    {
        parents_[a] = NOPARENT;
    }
    //printf("classify\n");

    std::vector<float> mi;
    getMutualInformationloc(xxyDist_.xyCounts, mi,inst);//计算局部I(xi;C)
    std::vector<CategoricalAttribute> order; //order存放的是所有的属性结点

    for (CategoricalAttribute a = 0; a < noCatAtts_; a++)
    {
        order.push_back(a);
    }
    //printf("i(xi|y)\n");
    if(!order.empty())
{
    miCmpClass cmp(&mi);
    std::sort(order.begin(), order.end(), cmp);

    crosstab<float> cmi = crosstab<float>(noCatAtts_);//I(x1,x2|y)
    getCondMutualInfloc(xxyDist_, cmi,inst);

    crosstab<float> cmixy = crosstab<float>(noCatAtts_);
    getCondMutualInflocas(xxyDist_,cmixy,inst);
    //printf("localcalcufin\n");
    // find the maximum spanning tree

    CategoricalAttribute firstAtt = order[0];

    parents_[firstAtt] = NOPARENT;

    float *maxWeight;   //最大加边权重
    float *maxin;  //最大加入权重
    std::vector<CategoricalAttribute> in;  //已经加入结构的结点
    in.push_back(firstAtt);
    CategoricalAttribute *bestSoFar;  //最优父节点
    CategoricalAttribute topCandidate = firstAtt;  //优先加入的节点
    std::set<CategoricalAttribute> available;    //剩余节点序列

    //safeAlloc(maxWeight, noCatAtts_);
    //safeAlloc(maxin, noCatAtts_);
    //safeAlloc(bestSoFar, noCatAtts_);
    maxWeight = new float[noCatAtts_];
    maxin = new float[noCatAtts_];
    bestSoFar = new unsigned int[noCatAtts_];

    maxWeight[firstAtt] = -std::numeric_limits<float>::max();
    maxin[firstAtt] = -std::numeric_limits<float>::max();

    for (CategoricalAttribute a = 1; a < order.size(); a++)
    {
        CategoricalAttribute b=order[a];
        maxin[b] = cmixy[b][firstAtt];
        maxWeight[b] = cmi[firstAtt][b];
        if (cmixy[b][firstAtt] > maxin[topCandidate])
            topCandidate = b;
        bestSoFar[b] = firstAtt;
        available.insert(b);
    }
    //printf("onein\n");

    while (!available.empty())
    {
        //const CategoricalAttribute current = topCandidate;//当前考虑的结点
        parents_[topCandidate] = bestSoFar[topCandidate];
        in.push_back(topCandidate);
        available.erase(topCandidate);
        float m=maxin[firstAtt];
        float n=maxWeight[firstAtt];

        if (!available.empty())
        {
            topCandidate = *available.begin();
            for (std::set<CategoricalAttribute>::const_iterator it =
                    available.begin(); it != available.end(); it++)
            {
                maxin[*it]=0;
                for(CategoricalAttribute i=0;i<in.size();i++)
                {
                    maxin[*it]+=cmixy[*it][in[i]];
                }
                if(maxin[*it]>m)
                {
                    m=maxin[*it];
                    topCandidate=*it;
                }
            }
            for(CategoricalAttribute i=0;i<in.size();i++)
            {
                if(cmi[topCandidate][in[i]]>n)
                {
                    n=cmi[topCandidate][in[i]];
                    bestSoFar[topCandidate]=in[i];
                }
            }
        }
    }
    in.clear();
    delete[] bestSoFar;
    delete[] maxWeight;
    delete[] maxin;
    available.clear();
}
    order.clear();

    for (CatValue y = 0; y < noClasses_; y++)
    {
        classDist[y] = xxyDist_.xyCounts.p(y);//类概率初始化
    }
    for (unsigned int x1 = 0; x1 < noCatAtts_; x1++)
    {
        const CategoricalAttribute parent = parents_[x1];

        if (parent == NOPARENT)
        {
            for (CatValue y = 0; y < noClasses_; y++)
            {
                classDist[y] *= xxyDist_.xyCounts.p(x1, inst.getCatVal(x1), y);//除类标签外无其他父节点
            }
        } else
        {

            for (CatValue y = 0; y < noClasses_; y++)
            {
                //std::cout<<x1<<" "<<parent<<std::endl;
                classDist[y] *= xxyDist_.p(x1, inst.getCatVal(x1), parent,
                        inst.getCatVal(parent), y);
            }
        }
    }
    normalise(classDist);//localtan

    //tan
    for (CatValue y = 0; y < noClasses_; y++)
    {
        tanclassDist[y] = xxyDist_.xyCounts.p(y);//类概率初始化
    }

    for (unsigned int x1 = 0; x1 < noCatAtts_; x1++)
    {
        const CategoricalAttribute parentt = tanparents_[x1];

        if (parentt == NOPARENT)
        {
            //printf("no\n");
            for (CatValue y = 0; y < noClasses_; y++)
            {
                tanclassDist[y] *= xxyDist_.xyCounts.p(x1, inst.getCatVal(x1), y);//除类标签外无其他父节点
            }
        } else
        {
            //std::cout<<parentt<<std::endl;
            for (CatValue y = 0; y < noClasses_; y++)
            {
                tanclassDist[y] *= xxyDist_.p(x1, inst.getCatVal(x1), parentt,
                        inst.getCatVal(parentt), y);
            }
        }
    }

    normalise(tanclassDist);


    mi.clear();
    parents_.clear();
    //联合概率结合
    for (int classno = 0; classno < noClasses_; classno++)
    {
        //printf("ccc\n");
        classDist[classno] =(classDist[classno]+tanclassDist[classno])/2;//两个集成
        //classDist[classno]=tanclassDist[classno];//tan
    }
    tanclassDist.clear();

}

void ltanas::finalisePass()
{
    assert(trainingIsFinished_ == false);

    std::vector<float> mi;
    getMutualInformation(xxyDist_.xyCounts, mi);//计算好了I(Xi;C)
    std::vector<CategoricalAttribute> order; //order存放的是所有的属性结点

    for (CategoricalAttribute a = 0; a < noCatAtts_; a++)
    {
        order.push_back(a);
    }
    if(!order.empty())
{
    miCmpClass cmp(&mi);
    std::sort(order.begin(), order.end(), cmp);

    crosstab<float> cmi = crosstab<float>(noCatAtts_);
    getCondMutualInf(xxyDist_, cmi);

    crosstab<float> cmixy = crosstab<float>(noCatAtts_);
    getXCondMutualInf(xxyDist_,cmixy);

    // find the maximum spanning tree

    CategoricalAttribute firstAtt = order[0];

    tanparents_[firstAtt] = NOPARENT;

    float *maxWeight;   //最大加边权重
    float *maxin;  //最大加入权重
    std::vector<CategoricalAttribute> in;  //已经加入结构的结点
    in.push_back(firstAtt);
    CategoricalAttribute *bestSoFar;  //最优父节点
    CategoricalAttribute topCandidate = firstAtt;  //优先加入的节点
    std::set<CategoricalAttribute> available;    //剩余节点序列

    safeAlloc(maxWeight, noCatAtts_);
    safeAlloc(maxin, noCatAtts_);
    safeAlloc(bestSoFar, noCatAtts_);

    maxWeight[firstAtt] = -std::numeric_limits<float>::max();
    maxin[firstAtt] = -std::numeric_limits<float>::max();

    for (CategoricalAttribute a = 1; a < noCatAtts_; a++)
    {
        CategoricalAttribute b=order[a];
        maxin[b] = cmixy[b][firstAtt];
        maxWeight[b] = cmi[firstAtt][b];
        if (cmixy[b][firstAtt] > maxin[topCandidate])
            topCandidate = b;
        bestSoFar[b] = firstAtt;
        available.insert(b);
    }

    while (!available.empty())
    {
        //const CategoricalAttribute current = topCandidate;//当前考虑的结点
        tanparents_[topCandidate] = bestSoFar[topCandidate];
        in.push_back(topCandidate);
        available.erase(topCandidate);
        float m=maxin[firstAtt];
        float n=maxWeight[firstAtt];

        if (!available.empty())
        {
            topCandidate = *available.begin();
            for (std::set<CategoricalAttribute>::const_iterator it =
                    available.begin(); it != available.end(); it++)
            {
                maxin[*it]=0;
                for(CategoricalAttribute i=0;i<in.size();i++)
                {
                    maxin[*it]+=cmixy[*it][in[i]];
                }
                if(maxin[*it]>m)
                {
                    m=maxin[*it];
                    topCandidate=*it;
                }
            }
            for(CategoricalAttribute i=0;i<in.size();i++)
            {
                if(cmi[topCandidate][in[i]]>n)
                {
                    n=cmi[topCandidate][in[i]];
                    bestSoFar[topCandidate]=in[i];
                }
            }
        }
    }
    in.clear();
    delete[] bestSoFar;
    delete[] maxWeight;
    delete[] maxin;
}
    order.clear();
    trainingIsFinished_ = true;
}

/// true iff no more passes are required. updated by finalisePass()

bool ltanas::trainingIsFinished()
{
    return trainingIsFinished_;
}
