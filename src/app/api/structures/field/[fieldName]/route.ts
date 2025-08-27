import { NextResponse } from 'next/server';
import path from 'path';
import fs from 'fs/promises';
import * as xlsx from 'xlsx';

export const dynamic = 'force-static';
export const revalidate = 0;

interface Structure {
    structure_name: string;
    file_path: string;
    wells: string[];
    wells_count: number;
    total_records: number;
    columns: string[];
    sample_data: Array<Record<string, unknown>>;
    error?: string;
}

interface FieldDetails {
    field_name: string;
    structures: Structure[];
    total_wells: number;
    total_records: number;
    all_wells: string[];
}

async function getFieldDetails(fieldName: string): Promise<FieldDetails> {
    const structuresDir = path.join(process.cwd(), 'data', 'structures');
    const fieldPath = path.join(structuresDir, fieldName);
    
    try {
        await fs.access(fieldPath);
    } catch {
        throw new Error(`Field not found: ${fieldName}`);
    }

    const structureFiles = (await fs.readdir(fieldPath))
        .filter(file => file.endsWith('.xlsx') && !file.startsWith('~'));

    const fieldDetails: FieldDetails = {
        field_name: fieldName,
        structures: [],
        total_wells: 0,
        total_records: 0,
        all_wells: []
    };

    const allWells = new Set<string>();

    for (const structureFile of structureFiles.sort()) {
        const structureName = structureFile.replace('.xlsx', '');
        const structurePath = path.join(fieldPath, structureFile);

        try {
            const file = await fs.readFile(structurePath);
            const workbook = xlsx.read(file, { type: 'buffer' });
            const sheetName = workbook.SheetNames[0];
            const worksheet = workbook.Sheets[sheetName];
            const data: unknown[][] = xlsx.utils.sheet_to_json(worksheet, { header: 1 });

            if (data.length === 0) {
                fieldDetails.structures.push({
                    structure_name: structureName,
                    file_path: structurePath,
                    wells: [],
                    wells_count: 0,
                    total_records: 0,
                    columns: [],
                    sample_data: []
                });
                continue;
            }

            const headers: string[] = data[0] ? data[0].map(h => String(h)) : [];
            const wellNameIndex = headers.indexOf('Well Name');
            const wells: string[] = [];

            if (wellNameIndex !== -1) {
                for (let i = 1; i < data.length; i++) {
                    const wellName = data[i][wellNameIndex];
                    if (wellName && !wells.includes(String(wellName))) {
                        wells.push(String(wellName));
                        allWells.add(String(wellName));
                    }
                }
            }

            // Get sample data (first 5 rows)
            const sampleData: Array<Record<string, unknown>> = data.slice(1, 6).map((row) => {
                const obj: Record<string, unknown> = {};
                headers.forEach((header, index) => {
                    obj[header] = (row as unknown[])[index];
                });
                return obj;
            });

            const structureInfo: Structure = {
                structure_name: structureName,
                file_path: structurePath,
                wells: wells.sort(),
                wells_count: wells.length,
                total_records: data.length - 1,
                columns: headers,
                sample_data: sampleData
            };

            fieldDetails.structures.push(structureInfo);
            fieldDetails.total_records += data.length - 1;

        } catch (e: unknown) {
            const message = e instanceof Error ? e.message : String(e);
            console.error(`Error reading ${structurePath}: ${message}`);
            fieldDetails.structures.push({
                structure_name: structureName,
                file_path: structurePath,
                wells: [],
                wells_count: 0,
                total_records: 0,
                columns: [],
                sample_data: [],
                error: message
            });
        }
    }

    fieldDetails.all_wells = Array.from(allWells).sort();
    fieldDetails.total_wells = fieldDetails.all_wells.length;

    return fieldDetails;
}

export async function GET(
    request: Request,
    { params }: { params: { fieldName: string } }
) {
    try {
        const data = await getFieldDetails(params.fieldName);
        return NextResponse.json(data);
    } catch (error: unknown) {
        const message = error instanceof Error ? error.message : String(error);
        return NextResponse.json({ error: message }, { status: 404 });
    }
}
